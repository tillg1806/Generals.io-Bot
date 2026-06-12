import argparse
import json
import multiprocessing as mp
import threading
import time
from datetime import datetime
from pathlib import Path

import socketio
from socketio import SimpleClient

from action_encoding import encode_action, legal_action_planes
from board_tensor import build_board_snapshot
from data_pipeline import (
    opponent_adjustment_for_names,
    process_public_game_result,
    process_self_play_batch,
)
from config import (
    BOT_ENDPOINT,
    BOT_EVENT_IDLE_TIMEOUT_SECONDS,
    BOT_KEY,
    GENERAL_GUESS_STATS_FILE,
    ROOM_ID,
    TRAINING_PROCESS_TIMEOUT_SECONDS,
    TRAINING_ARCHIVE_INTERVAL_MATCHES,
    TRAINING_COACH_INTERVAL_MATCHES,
    STARTUP_JAX_TRAINING_ENABLED,
    STARTUP_JAX_TRAINING_EPOCHS,
    STARTUP_JAX_TRAINING_PATIENCE,
    ACTION_SAMPLE_FIRST_MOVES,
    ACTION_SAMPLE_INTERVAL,
    ACTION_SAMPLE_MAX_PER_GAME,
    ACTION_VALUE_MIN_SAMPLES_TO_USE,
    POLICY_TENSOR_SAMPLE_FIRST_MOVES,
    POLICY_TENSOR_SAMPLE_INTERVAL,
    POLICY_TENSOR_SAMPLE_MAX_PER_GAME,
    TILE_FOG,
    TILE_FOG_OBSTACLE,
    TILE_MOUNTAIN,
    USER_ID,
    USERNAME,
)
from enemy_predictor import EnemyAttackPredictor
from game_state import GameState
from general_guesser import GeneralGuesser
from learning.jax_action_value_agent import JaxActionValueAgent
from learning.jax_agent import (
    train_action_value_agent,
    train_policy_agent,
    train_spawn_guess_agent,
)
from learning.jax_policy_agent import JaxPolicyAgent
from map_analyzer import analyze_state_map
from pathfinding import build_distance_map, distance_to_target
from power_management import WindowsSleepGuard
from replay_analyzer import BackgroundReplayAnalyzer
from self_play_dashboard import SelfPlayDashboard
from stats import ReserveTurnStats
from strategy import Strategy
from strategy_coach import StrategyCoach


REPLAY_INDEX_FILE = "data/replays/bot_replays.json"


class BotRunner:
    def __init__(
        self,
        room_id=ROOM_ID,
        user_id=USER_ID,
        username=USERNAME,
        stats_path=None,
        interactive=True,
        label=None,
        log_path=None,
        verbose_console=False,
        queue_mode="private",
        coach_path=None,
        prediction_path=None,
        suppress_console=False,
        use_spawn_grid_hint=False,
    ):
        self.client = SimpleClient()
        self.running = True
        self.game_started = False
        self.last_force = 0
        self.room_id = room_id
        self.user_id = user_id
        self.username = username
        self.interactive = interactive
        self.label = label or username
        self.log_path = Path(log_path) if log_path else None
        self.verbose_console = verbose_console
        self.queue_mode = queue_mode
        self.suppress_console = suppress_console
        self.use_spawn_grid_hint = use_spawn_grid_hint
        self.replay_id = None
        self.failure_reason = None
        self.last_event_time = time.time()
        self.enemy_general_guess_index = None
        self.enemy_general_guess_turn = None
        self.enemy_general_guess_confidence = None
        self.enemy_general_guess_reason = None
        self.enemy_general_guess_candidates = []
        self.enemy_general_beliefs = []
        self.enemy_general_actual_index = None
        self.guess_result_recorded = False
        self.map_metrics = None
        self.final_map_analysis = None
        self.move_count = 0
        self.action_samples = []
        self.policy_tensor_sample_count = 0
        self.game_start_data = {}
        self.player_names = []
        self.opponent_memory_adjustment = None
        self.jax_policy_agent = JaxPolicyAgent()
        self.jax_action_value_agent = JaxActionValueAgent(
            min_samples_to_use=ACTION_VALUE_MIN_SAMPLES_TO_USE,
        )
        self.jax_policy_adjustment = None
        self.last_jax_policy_turn = -1

        self.state = GameState()
        self.general_guesser = GeneralGuesser(force_spawn_grid=use_spawn_grid_hint)
        self.stats = ReserveTurnStats(stats_path) if stats_path else ReserveTurnStats()
        self.coach = StrategyCoach(path=coach_path or "data/profiles/strategy_profile.json", logger=self.log)
        self.enemy_predictor = EnemyAttackPredictor(path=prediction_path or "data/predictions/enemy_predictions.json")
        self.reserve_after_turn = self.stats.choose_value()
        self.city_focus_after_turn = self.stats.choose_city_focus_value()
        self.general_attack_after_turn = self.stats.choose_general_attack_value()
        self.strategy = Strategy(
            self.state,
            self.reserve_after_turn,
            self.city_focus_after_turn,
            self.general_attack_after_turn,
            logger=self.log,
        )
        self.strategy.action_value_agent = self.jax_action_value_agent
        self.coach.apply_start_profile(self.strategy)
        self.reserve_after_turn = self.strategy.reserve_after_turn
        self.city_focus_after_turn = self.strategy.city_focus_after_turn
        self.general_attack_after_turn = self.strategy.general_attack_after_turn

    def log(self, *parts, console=False):
        message = f"[{self.label}] " + " ".join(str(part) for part in parts)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as log_file:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"{timestamp} {message}\n")
        if (console or self.verbose_console) and not self.suppress_console:
            print(message, flush=True)

    def start_quit_listener(self):
        if not self.interactive:
            return

        def quit_listener():
            while True:
                if input().lower() == "q":
                    self.running = False
                    try:
                        self.client.emit("leave_game")
                        self.client.disconnect()
                    except Exception:
                        pass
                    break

        threading.Thread(target=quit_listener, daemon=True).start()

    def handle_game_update(self, data):
        turn = data.get("turn", -1)
        self.state.update(data)
        self.update_map_metrics()
        self.update_enemy_general_guess(turn)
        self.update_enemy_general_actual()
        self.update_map_metrics()
        self.enemy_predictor.observe(self.state, turn, self.replay_id)
        self.coach.observe(self.state, self.strategy, turn)
        self.apply_jax_policy(turn)

        self.log("My general index:", self.state.my_general_index)
        self.log("Enemy general index:", self.state.enemy_general_index)
        self.log("Visible enemy tiles:", len(self.state.visible_enemy_tiles))
        self.log("Tiles:", self.state.my_tile_count(), "/", self.state.biggest_enemy_tile_count())
        self.log("Map:", self.state.width, "x", self.state.height, "Turn:", turn)

        if turn == self.state.last_move_turn or self.state.my_general_index is None:
            return

        move = self.strategy.choose_move(turn)
        if move is None:
            return

        terrain, armies = self.state.split_map()
        self.client.emit("attack", (move.source, move.target, move.half))
        self.state.last_move_turn = turn
        self.move_count += 1
        self.record_action_sample(turn, move, terrain, armies)

        self.log(
            f"Move #{self.move_count}: {move.source} -> {move.target} | "
            f"Army: {armies[move.source]} | "
            f"Target-Terrain: {terrain[move.target]} | "
            f"Target: {move.strategy_target} | "
            f"Half: {move.half}",
            console=self.move_count <= 5 or self.move_count % 25 == 0,
        )

    def should_record_action_sample(self):
        if len(self.action_samples) >= ACTION_SAMPLE_MAX_PER_GAME:
            return False
        if self.move_count <= ACTION_SAMPLE_FIRST_MOVES:
            return True
        return ACTION_SAMPLE_INTERVAL > 0 and self.move_count % ACTION_SAMPLE_INTERVAL == 0

    def should_record_policy_tensor_sample(self):
        if self.policy_tensor_sample_count >= POLICY_TENSOR_SAMPLE_MAX_PER_GAME:
            return False
        if self.move_count <= POLICY_TENSOR_SAMPLE_FIRST_MOVES:
            return True
        return (
            POLICY_TENSOR_SAMPLE_INTERVAL > 0
            and self.move_count % POLICY_TENSOR_SAMPLE_INTERVAL == 0
        )

    def record_action_sample(self, turn, move, terrain, armies):
        if not self.should_record_action_sample():
            return

        selected = self.strategy.last_selected_move or {}
        score_components = selected.get("score_components") or {}
        source_army = armies[move.source] if move.source < len(armies) else None
        target_army = armies[move.target] if move.target < len(armies) else None
        target_terrain = terrain[move.target] if move.target < len(terrain) else None
        coach_snapshot = self.coach.last_snapshot
        action_index = encode_action(
            self.state.width,
            self.state.height,
            source=move.source,
            target=move.target,
            half=move.half,
        )
        include_policy_tensor = self.should_record_policy_tensor_sample()
        if include_policy_tensor:
            self.policy_tensor_sample_count += 1
        self.action_samples.append(
            {
                "move_number": self.move_count,
                "turn": turn,
                "visible_turn": self.state.visible_turn(turn),
                "source": move.source,
                "target": move.target,
                "half": move.half,
                "policy_action_index": action_index,
                "policy_legal_action_mask": (
                    legal_action_planes(
                        self.state,
                        terrain=terrain,
                        armies=armies,
                    )
                    if include_policy_tensor
                    else None
                ),
                "board": build_board_snapshot(
                    self.state,
                    terrain=terrain,
                    armies=armies,
                    include_tensor=include_policy_tensor,
                ),
                "strategy_target": move.strategy_target,
                "source_army": source_army,
                "target_army": target_army,
                "target_terrain": target_terrain,
                "target_distance": selected.get("target_distance"),
                "my_tiles": self.state.my_tile_count(),
                "enemy_tiles": self.state.biggest_enemy_tile_count(),
                "my_army": self.state.my_total_army(),
                "enemy_army": self.state.biggest_enemy_total_army(),
                "visible_enemy_tiles": len(self.state.visible_enemy_tiles),
                "seen_tiles": len(self.state.seen_tiles),
                "width": self.state.width,
                "height": self.state.height,
                "coach_mode": self.coach.last_mode,
                "coach_bias": dict(self.strategy.coach_bias),
                "coach_visible_my_cities": (
                    coach_snapshot.visible_my_cities if coach_snapshot else None
                ),
                "coach_visible_enemy_cities": (
                    coach_snapshot.visible_enemy_cities if coach_snapshot else None
                ),
                "score": selected.get("final_score", selected.get("score")),
                "base_score": selected.get("base_score"),
                "option": self.strategy.current_option.name,
                "option_reason": self.strategy.current_option.reason,
                "priority_reason": selected.get("priority_reason"),
                "lookahead_bonus": selected.get("lookahead_bonus", 0),
                "lookahead_reason": selected.get("lookahead_reason"),
                "action_value": selected.get("action_value"),
                "action_value_adjustment": selected.get("action_value_adjustment", 0),
                "score_components": score_components,
                "top_candidates": self.strategy.last_move_explanations,
                "flags": {
                    key: bool(selected.get(key))
                    for key in (
                        "target_is_new_tile",
                        "target_is_enemy_tile",
                        "target_is_enemy_general",
                        "can_take_city",
                        "follows_city_free_route",
                        "follows_fallback_route",
                        "opens_route_gateway",
                        "defends_general",
                        "attacks_threat",
                        "stalemate_breakout",
                        "stalemate_scout_push",
                    )
                },
            }
        )

    def run(self):
        self.start_quit_listener()

        try:
            self.client.connect(BOT_ENDPOINT, transports=["websocket"])
            self.log("Connected", console=True)
            if self.queue_mode == "private":
                self.log(f"Room: https://bot.generals.io/games/{self.room_id}", console=True)
            else:
                self.log(f"Queue: public {self.queue_mode}", console=True)
            self.log("Reserve ab sichtbarem Turn:", self.reserve_after_turn)
            self.log("City-Fokus ab sichtbarem Turn:", self.city_focus_after_turn)
            self.log("General-Angriff ab sichtbarem Turn:", self.general_attack_after_turn)

            self.client.emit("set_username", (self.user_id, self.username, BOT_KEY, None))
            set_username_response = self.client.receive(timeout=5)
            self.log("SET_USERNAME:", set_username_response, console=bool(set_username_response))
            if not self.can_continue_after_set_username(set_username_response):
                self.running = False
                return None

            self.join_game()
            self.last_event_time = time.time()

            while self.running:
                try:
                    if (
                        (self.game_started or self.queue_mode == "private")
                        and time.time() - self.last_event_time > BOT_EVENT_IDLE_TIMEOUT_SECONDS
                    ):
                        self.failure_reason = "event_idle_timeout"
                        self.log(
                            "No server events for",
                            BOT_EVENT_IDLE_TIMEOUT_SECONDS,
                            "seconds; stopping bot.",
                            console=True,
                        )
                        self.running = False
                        return None

                    if (
                        self.queue_mode == "private"
                        and not self.game_started
                        and time.time() - self.last_force > 2
                    ):
                        self.client.emit("set_force_start", (self.room_id, True))
                        self.last_force = time.time()

                    event = self.client.receive(timeout=5)
                    if event is None:
                        continue
                    self.last_event_time = time.time()

                    if event[0] != "game_update":
                        self.log("EVENT:", event)

                    if event[0] == "game_update":
                        self.handle_game_update(event[1])

                    if event[0] in ("gio_error", "error_set_username", "error_kicked"):
                        message = event[1] if len(event) > 1 else ""
                        self.failure_reason = f"{event[0]}:{message}"
                        self.log("SERVER ERROR:", event[0], message, console=True)
                        self.running = False
                        return None

                    if event[0] == "game_start":
                        data = event[1] if len(event) > 1 else {}
                        self.game_started = True
                        self.game_start_data = data
                        self.player_names = self.extract_player_names(data)
                        self.replay_id = data.get("replay_id")
                        self.state.start(data)
                        self.log("GAME START!", console=True)
                        self.log("My player index:", self.state.player_index)
                        if self.queue_mode == "1v1":
                            opponents = self.opponent_names()
                            self.log(
                                "Opponent:",
                                ", ".join(opponents) if opponents else "unknown",
                                console=True,
                            )
                        self.log("Replay:", self.replay_url(), console=True)
                        self.apply_opponent_memory()

                    elif event[0] in ("game_won", "game_lost"):
                        won = event[0] == "game_won"
                        if len(event) > 1 and isinstance(event[1], dict):
                            if "map" in event[1] or "map_diff" in event[1]:
                                self.state.update(event[1])
                                self.update_map_metrics()
                        self.update_enemy_general_actual_from_event(event)
                        self.update_enemy_general_actual()
                        self.update_final_map_analysis(event[0])
                        self.stats.record_result(self.reserve_after_turn, won)
                        self.stats.record_city_focus_result(self.city_focus_after_turn, won)
                        self.stats.record_general_attack_result(self.general_attack_after_turn, won)
                        self.coach.record_result(won, self)
                        self.enemy_predictor.flush(self.state, self.state.turn)
                        self.log("GAME END:", event[0], console=True)
                        self.log("Stats saved for reserve turn:", self.reserve_after_turn)
                        self.running = False
                        return won

                except socketio.exceptions.TimeoutError:
                    continue

        except Exception as e:
            self.log("Error:", repr(e), console=True)
            return None

        finally:
            try:
                self.client.disconnect()
            except Exception:
                pass

            self.log("Disconnected", console=True)

    def join_game(self):
        if self.queue_mode == "1v1":
            self.client.emit("join_1v1", (self.user_id, BOT_KEY, None, None, True))
            self.log("JOIN_1V1: public queue", console=True)
            return

        if self.queue_mode == "ffa":
            self.client.emit("play", (self.user_id, BOT_KEY, None, None))
            self.log("JOIN_FFA: public queue", console=True)
            return

        self.client.emit("join_private", (self.room_id, self.user_id, BOT_KEY, None))
        self.log("JOIN_PRIVATE:", self.client.receive(timeout=5), console=True)

    def replay_url(self):
        if not self.replay_id:
            return None

        return f"https://bot.generals.io/replays/{self.replay_id}"

    def update_enemy_general_guess(self, turn):
        if self.state.enemy_general_index is not None:
            self.enemy_general_beliefs = []
            self.strategy.set_enemy_general_beliefs([])
            return
        if self.state.width <= 0 or self.state.height <= 0:
            return
        if self.state.my_general_index is None:
            return

        terrain, _ = self.state.split_map()
        guess = self.general_guesser.choose(self.state, terrain, turn)
        if guess is None:
            self.enemy_general_guess_index = None
            self.enemy_general_guess_confidence = None
            self.enemy_general_guess_reason = None
            self.enemy_general_guess_candidates = []
            self.enemy_general_beliefs = []
            self.strategy.initial_enemy_general_guess = None
            self.strategy.set_enemy_general_beliefs([])
            return

        previous_guess = self.enemy_general_guess_index
        self.enemy_general_guess_index = guess.index
        self.enemy_general_guess_turn = turn
        self.enemy_general_guess_confidence = guess.confidence
        self.enemy_general_guess_reason = guess.reason
        self.enemy_general_guess_candidates = guess.candidates
        self.enemy_general_beliefs = guess.candidates
        self.strategy.set_enemy_general_beliefs(self.enemy_general_beliefs)
        if previous_guess == self.enemy_general_guess_index:
            return

        self.log(
            "Enemy general guess:",
            self.enemy_general_guess_index,
            "Confidence:",
            self.enemy_general_guess_confidence,
            "Beliefs:",
            [
                (candidate.get("index"), candidate.get("belief"))
                for candidate in self.enemy_general_beliefs[:3]
            ],
            "Grund:",
            self.enemy_general_guess_reason,
            "bei eigener Position",
            self.state.my_general_index,
            "Map:",
            f"{self.state.width}x{self.state.height}",
        )

    def update_enemy_general_actual(self):
        if self.state.enemy_general_index is not None:
            self.enemy_general_actual_index = self.state.enemy_general_index
            self.update_map_metrics()

    def update_enemy_general_actual_from_event(self, event):
        if len(event) < 2 or not isinstance(event[1], dict):
            return

        generals = event[1].get("generals")
        if not generals or self.state.player_index is None:
            return

        enemy_index = next(
            (
                index
                for index in range(len(generals))
                if index != self.state.player_index and generals[index] != -1
            ),
            None,
        )
        if enemy_index is not None:
            self.enemy_general_actual_index = generals[enemy_index]

    def general_guess_record(self, won):
        return {
            "actual_enemy_general_index": self.enemy_general_actual_index,
            "guess_correct": (
                self.enemy_general_guess_index is not None
                and self.enemy_general_actual_index is not None
                and self.enemy_general_guess_index == self.enemy_general_actual_index
            ),
            "guess_confidence": self.enemy_general_guess_confidence,
            "guess_reason": self.enemy_general_guess_reason,
            "guess_candidates": self.enemy_general_guess_candidates,
            "general_beliefs": self.enemy_general_beliefs,
            "guessed_enemy_general_index": self.enemy_general_guess_index,
            "guess_turn": self.enemy_general_guess_turn,
            "height": self.state.height,
            "spawn_grid_hint": {
                "enabled": self.use_spawn_grid_hint,
                "ready": self.general_guesser.spawn_grid_agent.is_ready(),
                "model_path": str(self.general_guesser.spawn_grid_agent.path),
            },
            "final_map_analysis": self.final_map_analysis,
            "map_metrics": self.map_metrics,
            "my_general_index": self.state.my_general_index,
            "replay_id": self.replay_id,
            "replay_url": self.replay_url(),
            "room_id": self.room_id,
            "status": "finished" if won is not None else "failed",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "width": self.state.width,
            "won": won,
        }

    def update_map_metrics(self):
        metrics = self.collect_map_metrics()
        if metrics:
            self.map_metrics = metrics

    def collect_map_metrics(self):
        if self.state.width <= 0 or self.state.height <= 0:
            return None

        terrain, _ = self.state.split_map()
        tile_count = self.state.width * self.state.height
        if len(terrain) < tile_count:
            return None

        mountain_count = sum(1 for tile in terrain[:tile_count] if tile == TILE_MOUNTAIN)
        fog_count = sum(1 for tile in terrain[:tile_count] if tile == TILE_FOG)
        fog_obstacle_count = sum(1 for tile in terrain[:tile_count] if tile == TILE_FOG_OBSTACLE)
        visible_tile_count = tile_count - fog_count - fog_obstacle_count
        passable_known_tile_count = sum(
            1
            for tile in terrain[:tile_count]
            if tile not in (TILE_MOUNTAIN, TILE_FOG_OBSTACLE)
        )
        city_count = len(set(city for city in self.state.cities if 0 <= city < tile_count))

        metrics = {
            "city_count": city_count,
            "city_density": city_count / tile_count if tile_count else 0,
            "fog_count": fog_count,
            "fog_obstacle_count": fog_obstacle_count,
            "height": self.state.height,
            "known_passable_tile_count": passable_known_tile_count,
            "mountain_count": mountain_count,
            "mountain_density": mountain_count / tile_count if tile_count else 0,
            "tile_count": tile_count,
            "visible_tile_count": visible_tile_count,
            "width": self.state.width,
        }

        metrics["guess_distance"] = self.general_distance_metrics(
            self.enemy_general_guess_index,
            terrain,
        )
        metrics["actual_general_distance"] = self.general_distance_metrics(
            self.enemy_general_actual_index,
            terrain,
        )
        return metrics

    def general_distance_metrics(self, target, terrain):
        if self.state.my_general_index is None or target is None:
            return None
        if target < 0 or target >= self.state.width * self.state.height:
            return None

        distances = build_distance_map(
            self.state,
            target,
            terrain,
            avoid_cities=False,
        )
        path_distance = distances.get(self.state.my_general_index)

        return {
            "manhattan": distance_to_target(
                self.state,
                self.state.my_general_index,
                target,
            ),
            "path": path_distance,
            "target_index": target,
        }

    def update_final_map_analysis(self, status):
        self.final_map_analysis = analyze_state_map(
            self.state,
            self.state.map_data,
            status=status,
        )
        if self.final_map_analysis:
            visibility = self.final_map_analysis.get("visibility") or {}
            self.log(
                "Final map analysis:",
                "visible =",
                visibility.get("visible_ratio"),
                "complete =",
                visibility.get("is_full_map_visible"),
                "symmetry =",
                self.final_map_analysis.get("symmetry"),
            )

    def extract_player_names(self, data):
        for key in ("usernames", "users", "player_names", "players"):
            value = data.get(key)
            if isinstance(value, list):
                names = []
                for item in value:
                    if isinstance(item, str):
                        names.append(item)
                    elif isinstance(item, dict):
                        names.append(
                            item.get("username")
                            or item.get("name")
                            or item.get("id")
                            or "unknown"
                        )
                    else:
                        names.append(str(item))
                return names
        return []

    def opponent_names(self):
        if not self.player_names:
            return []

        if self.state.player_index is None:
            return [name for name in self.player_names if name != self.username]

        return [
            name
            for index, name in enumerate(self.player_names)
            if index != self.state.player_index
        ]

    def apply_opponent_memory(self):
        adjustment = opponent_adjustment_for_names(self.opponent_names())
        if not adjustment:
            return

        self.opponent_memory_adjustment = adjustment
        for key, delta in adjustment.get("bias", {}).items():
            current = float(self.strategy.coach_bias.get(key, 0.0))
            new_value = round(max(-2.0, min(3.0, current + float(delta))), 3)
            self.strategy.coach_bias[key] = new_value
            self.coach.opponent_bias[key] = round(
                max(-2.0, min(3.0, float(self.coach.opponent_bias.get(key, 0.0)) + float(delta))),
                3,
            )

        timing = adjustment.get("timing", {})
        self.reserve_after_turn = max(
            20,
            min(600, self.reserve_after_turn + int(timing.get("reserve_delta", 0))),
        )
        self.city_focus_after_turn = max(
            30,
            min(240, self.city_focus_after_turn + int(timing.get("city_focus_delta", 0))),
        )
        self.general_attack_after_turn = max(
            60,
            min(280, self.general_attack_after_turn + int(timing.get("general_attack_delta", 0))),
        )
        self.strategy.reserve_after_turn = self.reserve_after_turn
        self.strategy.city_focus_after_turn = self.city_focus_after_turn
        self.strategy.general_attack_after_turn = self.general_attack_after_turn
        self.log(
            "Opponent memory applied:",
            adjustment.get("reasons"),
            "bias",
            adjustment.get("bias"),
            "timing",
            timing,
            console=True,
        )

    def apply_jax_policy(self, turn):
        if not self.jax_policy_agent.is_ready():
            return

        visible_turn = self.state.visible_turn(turn)
        if visible_turn < 25:
            return
        if visible_turn - self.last_jax_policy_turn < 20:
            return

        adjustment = self.jax_policy_agent.recommend_for_strategy(self.strategy, self.coach)
        self.last_jax_policy_turn = visible_turn
        if not adjustment or not adjustment.get("bias"):
            return

        self.jax_policy_adjustment = adjustment
        for key in self.coach.model_bias:
            self.coach.model_bias[key] = 0.0

        for key, delta in adjustment.get("bias", {}).items():
            current = float(self.strategy.coach_bias.get(key, 0.0))
            self.strategy.coach_bias[key] = round(max(-2.0, min(3.0, current + float(delta))), 3)
            self.coach.model_bias[key] = round(max(-2.0, min(3.0, float(delta))), 3)

        self.log(
            "JAX policy adjustment:",
            adjustment.get("reason"),
            "win_probability",
            adjustment.get("win_probability"),
            "bias",
            adjustment.get("bias"),
        )

    def can_continue_after_set_username(self, response):
        if not response:
            return True
        if response[0] != "error_set_username":
            return True

        message = response[1] if len(response) > 1 else ""
        if "already have a username" in message:
            return True

        if not message:
            self.failure_reason = "account_creation_rate_limited"
        elif "wait a bit longer" in message:
            self.failure_reason = "account_creation_rate_limited"
        elif "cannot start with [Bot]" in message:
            self.failure_reason = "username_bot_prefix_forbidden"
        elif "Username too long" in message:
            self.failure_reason = "username_too_long"
        elif "username is taken" in message:
            self.failure_reason = "username_taken"
        else:
            self.failure_reason = "set_username_failed"

        self.log("Username could not be set; bot will not start.")
        return False


def write_bot_result(result_path, runner, won):
    if result_path is None:
        return

    status = "finished"
    if won is None:
        status = "failed" if runner.failure_reason else "unknown"

    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "room_id": runner.room_id,
                "user_id": runner.user_id,
                "username": runner.username,
                "label": runner.label,
                "player_names": runner.player_names,
                "opponent_names": runner.opponent_names(),
                "opponent_memory_adjustment": runner.opponent_memory_adjustment,
                "jax_policy_adjustment": runner.jax_policy_adjustment,
                "spawn_grid_hint": {
                    "enabled": runner.use_spawn_grid_hint,
                    "ready": runner.general_guesser.spawn_grid_agent.is_ready(),
                    "model_path": str(runner.general_guesser.spawn_grid_agent.path),
                },
                "log_path": str(runner.log_path) if runner.log_path else None,
                "general_guess": runner.general_guess_record(won),
                "action_sample_count": len(runner.action_samples),
                "action_samples": runner.action_samples,
                "action_value_model": {
                    "ready": runner.jax_action_value_agent.is_ready(),
                    "sample_count": runner.jax_action_value_agent.sample_count(),
                    "min_samples_to_use": ACTION_VALUE_MIN_SAMPLES_TO_USE,
                },
                "last_move_explanations": runner.strategy.last_move_explanations,
                "strategy_option": {
                    "name": runner.strategy.current_option.name,
                    "reason": runner.strategy.current_option.reason,
                    "started_at_visible_turn": runner.strategy.current_option.started_at_visible_turn,
                },
                "final_map_analysis": runner.final_map_analysis,
                "won": won,
                "reserve_after_turn": runner.reserve_after_turn,
                "city_focus_after_turn": runner.city_focus_after_turn,
                "general_attack_after_turn": runner.general_attack_after_turn,
                "coach_mode": runner.coach.last_mode,
                "coach_bias": runner.strategy.coach_bias,
                "coach_model_bias": runner.coach.model_bias,
                "coach_events": runner.coach.game_events,
                "coach_visible_my_cities": (
                    runner.coach.last_snapshot.visible_my_cities
                    if runner.coach.last_snapshot
                    else None
                ),
                "coach_visible_enemy_cities": (
                    runner.coach.last_snapshot.visible_enemy_cities
                    if runner.coach.last_snapshot
                    else None
                ),
                "coach_suspected_enemy_city_advantage": (
                    runner.coach.last_snapshot.suspected_enemy_city_advantage
                    if runner.coach.last_snapshot
                    else None
                ),
                "stalemate": {
                    "active_at_finish": runner.strategy.stalemate.active,
                    "activated_at_visible_turn": runner.strategy.stalemate.activated_at_visible_turn,
                    "last_progress_visible_turn": runner.strategy.stalemate.last_progress_visible_turn,
                    "repeated_target_count": runner.strategy.stalemate.repeated_target_count,
                    "reason": runner.strategy.stalemate.reason,
                },
                "replay_id": runner.replay_id,
                "replay_url": runner.replay_url(),
                "failure_reason": runner.failure_reason,
                "status": status,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def add_replay_record(runner, won, path=REPLAY_INDEX_FILE):
    if not runner.replay_id:
        return

    replay_path = Path(path)
    try:
        records = json.loads(replay_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        records = []

    if not isinstance(records, list):
        records = []

    existing = {
        record.get("replay_id"): record
        for record in records
        if isinstance(record, dict) and record.get("replay_id")
    }
    status = "won" if won is True else "lost" if won is False else "unknown"
    existing[runner.replay_id] = {
        "replay_id": runner.replay_id,
        "replay_url": runner.replay_url(),
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "queue_mode": runner.queue_mode,
        "username": runner.username,
        "player_names": runner.player_names,
        "opponent_names": runner.opponent_names(),
        "opponent_memory_adjustment": runner.opponent_memory_adjustment,
        "jax_policy_adjustment": runner.jax_policy_adjustment,
        "player_index": runner.state.player_index,
        "turn": getattr(runner.state, "turn", runner.state.last_move_turn),
        "move_count": runner.move_count,
        "width": runner.state.width,
        "height": runner.state.height,
        "my_general_index": runner.state.my_general_index,
        "enemy_general_index": runner.state.enemy_general_index,
        "final_map_analysis": runner.final_map_analysis,
        "failure_reason": runner.failure_reason,
        "coach_mode": runner.coach.last_mode,
        "coach_bias": runner.strategy.coach_bias,
        "coach_model_bias": runner.coach.model_bias,
        "coach_visible_my_cities": (
            runner.coach.last_snapshot.visible_my_cities
            if runner.coach.last_snapshot
            else None
        ),
        "coach_visible_enemy_cities": (
            runner.coach.last_snapshot.visible_enemy_cities
            if runner.coach.last_snapshot
            else None
        ),
        "coach_suspected_enemy_city_advantage": (
            runner.coach.last_snapshot.suspected_enemy_city_advantage
            if runner.coach.last_snapshot
            else None
        ),
    }

    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(
        json.dumps(
            sorted(
                existing.values(),
                key=lambda record: record.get("updated_at", ""),
                reverse=True,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def run_training_bot(
    room_id,
    user_id,
    username,
    stats_path,
    result_path,
    label,
    log_path=None,
    coach_path=None,
    prediction_path=None,
    use_spawn_grid_hint=False,
):
    runner = BotRunner(
        room_id=room_id,
        user_id=user_id,
        username=username,
        stats_path=stats_path,
        interactive=False,
        label=label,
        log_path=log_path,
        coach_path=coach_path,
        prediction_path=prediction_path,
        suppress_console=bool(log_path),
        use_spawn_grid_hint=use_spawn_grid_hint,
    )
    won = runner.run()
    write_bot_result(result_path, runner, won)


def write_pending_bot_result(
    result_path,
    room_id,
    user_id,
    username,
    label,
    log_path=None,
    use_spawn_grid_hint=False,
):
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "room_id": room_id,
                "user_id": user_id,
                "username": username,
                "label": label,
                "player_names": [],
                "opponent_names": [],
                "opponent_memory_adjustment": None,
                "jax_policy_adjustment": None,
                "spawn_grid_hint": {
                    "enabled": use_spawn_grid_hint,
                    "ready": None,
                    "model_path": None,
                },
                "log_path": str(log_path) if log_path else None,
                "general_guess": None,
                "final_map_analysis": None,
                "won": None,
                "reserve_after_turn": None,
                "city_focus_after_turn": None,
                "general_attack_after_turn": None,
                "coach_mode": None,
                "coach_bias": None,
                "coach_model_bias": None,
                "coach_events": [],
                "coach_visible_my_cities": None,
                "coach_visible_enemy_cities": None,
                "coach_suspected_enemy_city_advantage": None,
                "replay_id": None,
                "replay_url": None,
                "failure_reason": None,
                "status": "running",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def mark_training_process_failed(result_path, reason):
    path = Path(result_path)
    if not path.exists():
        return

    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    if result.get("status") != "running":
        return

    result["failure_reason"] = reason
    result["status"] = "failed"
    result["won"] = None
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def mark_training_process_timeout(result_path):
    mark_training_process_failed(result_path, "process_timeout")


def read_json_file(path, fallback):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def write_json_file(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def general_guess_record_key(record):
    replay_id = record.get("replay_id")
    if replay_id:
        return (
            replay_id,
            record.get("room_id"),
            record.get("my_general_index"),
            record.get("guessed_enemy_general_index"),
        )

    return (
        record.get("timestamp"),
        record.get("room_id"),
        record.get("my_general_index"),
        record.get("guessed_enemy_general_index"),
    )


def add_general_guess_records(records, stats_path=GENERAL_GUESS_STATS_FILE):
    data = read_json_file(stats_path, {"records": [], "summary": {}})
    existing_records = data.get("records", [])
    seen = {general_guess_record_key(record) for record in existing_records}

    for record in records:
        if not record:
            continue
        if record.get("guessed_enemy_general_index") is None:
            continue

        key = general_guess_record_key(record)
        if key in seen:
            continue

        existing_records.append(record)
        seen.add(key)

    data = {
        "records": existing_records,
        "summary": build_general_guess_summary(existing_records),
    }
    write_json_file(stats_path, data)


def build_general_guess_summary(records):
    by_setup = {}
    by_map_size = {}
    for record in records:
        map_metrics = record.get("map_metrics") or {}
        setup_key = "|".join(
            str(part)
            for part in (
                record.get("width"),
                record.get("height"),
                record.get("my_general_index"),
                record.get("guessed_enemy_general_index"),
                record.get("actual_enemy_general_index"),
            )
        )
        entry = by_setup.setdefault(
            setup_key,
            {
                "actual_enemy_general_index": record.get("actual_enemy_general_index"),
                "games": 0,
                "guessed_enemy_general_index": record.get("guessed_enemy_general_index"),
                "height": record.get("height"),
                "my_general_index": record.get("my_general_index"),
                "wins": 0,
                "width": record.get("width"),
            },
        )
        entry["games"] += 1
        if record.get("won"):
            entry["wins"] += 1

        map_key = f"{record.get('width')}x{record.get('height')}"
        map_entry = by_map_size.setdefault(
            map_key,
            {
                "actual_path_distance_total": 0,
                "actual_path_distance_with_value": 0,
                "city_count_total": 0,
                "games": 0,
                "guess_path_distance_total": 0,
                "guess_path_distance_with_value": 0,
                "height": record.get("height"),
                "mountain_count_total": 0,
                "tile_count": map_metrics.get("tile_count"),
                "width": record.get("width"),
            },
        )
        map_entry["games"] += 1
        map_entry["city_count_total"] += map_metrics.get("city_count") or 0
        map_entry["mountain_count_total"] += map_metrics.get("mountain_count") or 0

        actual_distance = map_metrics.get("actual_general_distance") or {}
        actual_path = actual_distance.get("path")
        if actual_path is not None:
            map_entry["actual_path_distance_total"] += actual_path
            map_entry["actual_path_distance_with_value"] += 1

        guess_distance = map_metrics.get("guess_distance") or {}
        guess_path = guess_distance.get("path")
        if guess_path is not None:
            map_entry["guess_path_distance_total"] += guess_path
            map_entry["guess_path_distance_with_value"] += 1

    for map_entry in by_map_size.values():
        games = map_entry["games"]
        map_entry["avg_city_count"] = map_entry["city_count_total"] / games if games else 0
        map_entry["avg_mountain_count"] = map_entry["mountain_count_total"] / games if games else 0

        actual_count = map_entry["actual_path_distance_with_value"]
        map_entry["avg_actual_path_distance"] = (
            map_entry["actual_path_distance_total"] / actual_count
            if actual_count
            else None
        )

        guess_count = map_entry["guess_path_distance_with_value"]
        map_entry["avg_guess_path_distance"] = (
            map_entry["guess_path_distance_total"] / guess_count
            if guess_count
            else None
        )

    return {
        "by_map_size": by_map_size,
        "by_setup": by_setup,
        "correct_guesses": sum(1 for record in records if record.get("guess_correct")),
        "records": len(records),
        "with_actual": sum(
            1
            for record in records
            if record.get("actual_enemy_general_index") is not None
        ),
    }


def collect_general_guess_records(result_paths):
    records = []
    for result_path in result_paths:
        result = read_json_file(result_path, None)
        if isinstance(result, dict):
            records.append(result.get("general_guess"))

    return records


def collect_replay_results(result_paths):
    results = []

    for result_path in result_paths:
        path = Path(result_path)
        if not path.exists():
            continue

        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue

    matches = {}
    for result in results:
        room_id = result.get("room_id")
        match = matches.setdefault(
            room_id,
            {
                "room_id": room_id,
                "replay_id": result.get("replay_id"),
                "replay_url": result.get("replay_url"),
                "bots": [],
            },
        )
        if not match.get("replay_id") and result.get("replay_id"):
            match["replay_id"] = result.get("replay_id")
            match["replay_url"] = result.get("replay_url")

        match["bots"].append(result)

    return {
        "matches": list(matches.values()),
        "bots": results,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--log-files",
        action="store_true",
        help="Writes detailed bot logs to files instead of the console.",
    )
    parser.add_argument(
        "-q",
        "--queue",
        choices=("private", "1v1", "ffa"),
        default="private",
        help="Selects the game mode: private custom rooms, public 1v1, or public FFA.",
    )
    parser.add_argument(
        "--self-play",
        action="store_true",
        help="Starts endless local self-play with strategy variants.",
    )
    parser.add_argument(
        "--parallel-games",
        type=int,
        default=1,
        help="Number of self-play games to run in parallel.",
    )
    parser.add_argument(
        "--self-play-start-delay",
        type=float,
        default=2.0,
        help="Seconds to wait after dispatching one self-play bot pair before dispatching the next pair.",
    )
    parser.add_argument(
        "--self-play-requeue-delay",
        type=float,
        default=20.0,
        help="Seconds an idle self-play bot pair waits before it can be dispatched again.",
    )
    parser.add_argument(
        "--self-play-run-id",
        default=None,
        help="Optional fixed self-play run id, useful for external log consoles.",
    )
    parser.add_argument(
        "--training-archive-interval",
        type=int,
        default=TRAINING_ARCHIVE_INTERVAL_MATCHES,
        help="Number of finished self-play matches between raw training ZIP archives.",
    )
    parser.add_argument(
        "--training-coach-interval",
        type=int,
        default=TRAINING_COACH_INTERVAL_MATCHES,
        help="Number of finished self-play matches between coach checkpoints and JAX training runs.",
    )
    parser.add_argument(
        "--disable-replay-analyzer",
        action="store_true",
        help="Disables the low-priority self-play replay metadata analyzer.",
    )
    parser.add_argument(
        "--replay-analyzer-interval",
        type=float,
        default=15.0,
        help="Seconds between background replay metadata analyzer passes.",
    )
    parser.add_argument(
        "--train-jax-agent",
        action="store_true",
        help="Trains the JAX policy agent from data/training/policy_samples.jsonl.",
    )
    parser.add_argument(
        "--train-action-value-agent",
        action="store_true",
        help="Trains the JAX action-value agent from data/training/action_samples.jsonl.",
    )
    parser.add_argument(
        "--train-spawn-guess-agent",
        action="store_true",
        help="Trains the JAX spawn guess agent from data/replays/duel_map_dataset.jsonl.",
    )
    parser.add_argument(
        "--skip-startup-training",
        action="store_true",
        help="Skips the automatic short JAX refresh before a normal bot/simulator/self-play start.",
    )
    parser.add_argument(
        "--test-spawn-grid-hint",
        action="store_true",
        help="Temporarily enables the experimental JAX spawn grid hint without editing the model file.",
    )
    parser.add_argument(
        "--spawn-grid-ab-test",
        action="store_true",
        help="In self-play, runs one spawn-grid-hint bot against one baseline bot per match.",
    )
    parser.add_argument(
        "--sim-self-play",
        action="store_true",
        help="Runs local simulator games with the strategy adapter instead of live generals.io rooms.",
    )
    parser.add_argument(
        "--sim-games",
        type=int,
        default=32,
        help="Number of local simulator games to run.",
    )
    parser.add_argument(
        "--sim-parallel-games",
        type=int,
        default=16,
        help="Number of local simulator games to keep active in the Python adapter loop.",
    )
    parser.add_argument(
        "--sim-grid-size",
        type=int,
        default=10,
        help="Square grid size for local simulator games.",
    )
    parser.add_argument(
        "--sim-truncation",
        type=int,
        default=500,
        help="Maximum simulator steps per game.",
    )
    parser.add_argument(
        "--sim-opponent",
        choices=("random", "expander", "hunter"),
        default="expander",
        help="Built-in simulator opponent for local games.",
    )
    parser.add_argument(
        "--sim-path",
        default=None,
        help="Path to generals-bots; defaults to GENERALS_BOTS_PATH or ../Generals-Bot-Sim/generals-bots.",
    )
    parser.add_argument(
        "--no-requeue",
        action="store_true",
        help="Runs only one live bot game and then stops instead of searching again.",
    )
    return parser.parse_args()


def stop_runner_now(runner):
    if runner is None:
        return

    runner.running = False
    try:
        runner.client.emit("leave_game")
        runner.client.disconnect()
    except Exception:
        pass


def start_auto_game_listener(
    stop_after_current,
    current_runner,
    runner_lock,
    allow_force_stop=True,
    force_stop_event=None,
):
    def handle_command(command):
        if command == "e":
            stop_after_current.set()
            print("Autostart stopped: current game will finish.", flush=True)
            return True

        if command == "q":
            stop_after_current.set()
            if not allow_force_stop:
                print("Autostart stopped: currently running matches will finish.", flush=True)
                return True

            if force_stop_event is not None:
                force_stop_event.set()

            with runner_lock:
                runner = current_runner.get("runner")
            stop_runner_now(runner)
            print("Immediate stop requested.", flush=True)
            return True

        return False

    def listener():
        try:
            import msvcrt

            while not stop_after_current.is_set():
                if not msvcrt.kbhit():
                    time.sleep(0.1)
                    continue

                command = msvcrt.getwch().strip().lower()
                if handle_command(command):
                    break
            return
        except ImportError:
            pass

        while not stop_after_current.is_set():
            try:
                command = input().strip().lower()
            except EOFError:
                break

            if handle_command(command):
                break

    threading.Thread(target=listener, daemon=True).start()


def run_auto_games(write_log_files=False, queue_mode="private", use_spawn_grid_hint=False, requeue=True):
    stop_after_current = threading.Event()
    current_runner = {"runner": None}
    runner_lock = threading.Lock()
    game_number = 1
    run_id = f"public_{queue_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs/public_1v1") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Auto mode active ({queue_mode}, requeue={'on' if requeue else 'off'}). "
        "Input: e = stop after current game, q = stop immediately.",
        flush=True,
    )
    print(f"Public game results: {run_dir}", flush=True)
    if write_log_files:
        print("Detailed logs: logs/public_games", flush=True)
    start_auto_game_listener(stop_after_current, current_runner, runner_lock)

    while not stop_after_current.is_set():
        log_path = None
        if write_log_files:
            log_path = Path("logs/public_games") / f"game_{game_number:03d}.log"

        runner = BotRunner(
            interactive=False,
            label=f"game-{game_number}",
            log_path=log_path,
            queue_mode=queue_mode,
            use_spawn_grid_hint=use_spawn_grid_hint,
        )
        with runner_lock:
            current_runner["runner"] = runner

        print(f"Starting game {game_number}.", flush=True)
        with WindowsSleepGuard("auto game"):
            won = runner.run()
        result_path = run_dir / f"game_{game_number:03d}_result.json"
        write_bot_result(result_path, runner, won)
        add_general_guess_records([runner.general_guess_record(won)])
        add_replay_record(runner, won)
        training_summary = process_public_game_result(result_path)
        if training_summary.get("processed"):
            print(
                "Training data:",
                f"+{training_summary['action_samples_written']} action samples,",
                f"+{training_summary['policy_samples_written']} policy samples.",
                flush=True,
            )

        with runner_lock:
            current_runner["runner"] = None

        if stop_after_current.is_set():
            break

        if not requeue:
            print("Requeue disabled; stopping after this game.", flush=True)
            break

        if won is None:
            if runner.failure_reason and (
                "username" in runner.failure_reason
                or "gio_error" in runner.failure_reason
                or "error_set_username" in runner.failure_reason
                or "Account Disabled" in runner.failure_reason
                or runner.failure_reason
                in (
                    "account_creation_rate_limited",
                    "username_bot_prefix_forbidden",
                    "username_too_long",
                    "username_taken",
                    "set_username_failed",
                )
            ):
                print(
                    "Stopping because of account/username error:",
                    runner.failure_reason,
                    flush=True,
                )
                break

            print("Game did not finish normally; searching again.", flush=True)
            game_number += 1
            time.sleep(2)
            continue

        game_number += 1
        time.sleep(2)

    print("No new games will be started.", flush=True)


SELF_PLAY_VARIANTS = [
    {
        "name": "expand",
        "learned": {
            "expansion_bias": 0.7,
            "city_bias": 0.1,
            "attack_bias": 0.0,
            "defense_bias": 0.0,
            "route_bias": 0.3,
            "reserve_delta": 10,
            "city_focus_delta": -10,
            "general_attack_delta": 10,
        },
    },
    {
        "name": "city",
        "learned": {
            "expansion_bias": 0.1,
            "city_bias": 0.8,
            "attack_bias": 0.1,
            "defense_bias": 0.0,
            "route_bias": 0.3,
            "reserve_delta": 0,
            "city_focus_delta": -35,
            "general_attack_delta": 10,
        },
    },
    {
        "name": "attack",
        "learned": {
            "expansion_bias": 0.2,
            "city_bias": 0.1,
            "attack_bias": 0.8,
            "defense_bias": 0.0,
            "route_bias": 0.6,
            "reserve_delta": -20,
            "city_focus_delta": 0,
            "general_attack_delta": -45,
        },
    },
    {
        "name": "defense",
        "learned": {
            "expansion_bias": 0.1,
            "city_bias": 0.2,
            "attack_bias": -0.1,
            "defense_bias": 0.8,
            "route_bias": 0.1,
            "reserve_delta": 35,
            "city_focus_delta": 10,
            "general_attack_delta": 20,
        },
    },
]


def ensure_self_play_profile(path, variant):
    profile_path = Path(path)
    if profile_path.exists():
        return

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(
            {
                "version": 1,
                "variant": variant["name"],
                "games": 0,
                "wins": 0,
                "losses": 0,
                "learned": variant["learned"],
                "recent_games": [],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

SELF_PLAY_ACCOUNTS = [
    {
        "user_id": f"{USER_ID}_self_01",
        "username": f"{USERNAME}_Self_01",
    },
    {
        "user_id": f"{USER_ID}_self_02",
        "username": f"{USERNAME}_Self_02",
    },
    {
        "user_id": f"{USER_ID}_self_03",
        "username": f"{USERNAME}_Self_03",
    },
    {
        "user_id": f"{USER_ID}_self_04",
        "username": f"{USERNAME}_Self_04",
    },
    {
        "user_id": f"{USER_ID}_self_05",
        "username": f"{USERNAME}_Self_05",
    },
    {
        "user_id": f"{USER_ID}_self_06",
        "username": f"{USERNAME}_Self_06",
    },
    {
        "user_id": f"{USER_ID}_self_07",
        "username": f"{USERNAME}_Self_07",
    },
    {
        "user_id": f"{USER_ID}_self_08",
        "username": f"{USERNAME}_Self_08",
    },
    {
        "user_id": f"{USER_ID}_self_09",
        "username": f"{USERNAME}_Self_09",
    },
    {
        "user_id": f"{USER_ID}_self_10",
        "username": f"{USERNAME}_Self_10",
    },
]

def self_play_accounts(slot_number):
    start_index = (slot_number - 1) * 2
    end_index = start_index + 2

    if end_index > len(SELF_PLAY_ACCOUNTS):
        raise ValueError(
            f"Self-play slot {slot_number} needs bots {start_index + 1}-{end_index}, "
            f"but only {len(SELF_PLAY_ACCOUNTS)} fixed accounts are configured."
        )

    return SELF_PLAY_ACCOUNTS[start_index:end_index]

def start_self_play_match(
    match_number,
    slot_number,
    run_id,
    run_dir,
    write_log_files,
    use_spawn_grid_hint=False,
    spawn_grid_ab_test=False,
):
    variant_a = SELF_PLAY_VARIANTS[(match_number - 1) % len(SELF_PLAY_VARIANTS)]
    variant_b = SELF_PLAY_VARIANTS[(match_number) % len(SELF_PLAY_VARIANTS)]
    accounts = self_play_accounts(slot_number)
    room_id = f"{ROOM_ID}_self_{run_id}_{match_number:04d}"

    result_paths = []
    stats_paths = []
    processes = []

    for seat, variant in enumerate((variant_a, variant_b)):
        seat_uses_spawn_grid = use_spawn_grid_hint
        if spawn_grid_ab_test:
            hint_seat = (match_number + slot_number) % 2
            seat_uses_spawn_grid = seat == hint_seat

        hint_suffix = "-gridhint" if seat_uses_spawn_grid else "-baseline"
        label = f"self-{match_number:04d}-{seat}-{variant['name']}{hint_suffix}"
        stats_path = run_dir / f"{label}_stats.json"
        result_path = run_dir / f"{label}_result.json"
        log_path = run_dir / "logs" / "slots" / f"slot_{slot_number}.log" if write_log_files else None
        coach_path = run_dir / "profiles" / f"{variant['name']}.json"
        prediction_path = run_dir / "predictions" / f"{label}_enemy_predictions.json"

        ensure_self_play_profile(coach_path, variant)
        if seat == 0 and log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    "\n"
                    f"===== MATCH {match_number:04d} | SLOT {slot_number} | "
                    f"{variant_a['name']} vs {variant_b['name']} =====\n"
                    f"Room: https://bot.generals.io/games/{room_id}\n"
                    f"Seat 0: {variant_a['name']} "
                    f"{'gridhint' if (use_spawn_grid_hint or (spawn_grid_ab_test and hint_seat == 0)) else 'baseline'}\n"
                    f"Seat 1: {variant_b['name']} "
                    f"{'gridhint' if (use_spawn_grid_hint or (spawn_grid_ab_test and hint_seat == 1)) else 'baseline'}\n"
                )

        stats_paths.append(stats_path)
        result_paths.append(result_path)

        write_pending_bot_result(
            result_path,
            room_id,
            accounts[seat]["user_id"],
            accounts[seat]["username"],
            label,
            log_path,
            seat_uses_spawn_grid,
        )

        process = mp.Process(
            target=run_training_bot,
            args=(
                room_id,
                accounts[seat]["user_id"],
                accounts[seat]["username"],
                str(stats_path),
                str(result_path),
                label,
                str(log_path) if log_path else None,
                str(coach_path),
                str(prediction_path),
                seat_uses_spawn_grid,
            ),
        )
        process.start()

        processes.append(
            {
                "process": process,
                "result_path": result_path,
                "started_at": time.time(),
                "label": label,
            }
        )

    return {
        "match_number": match_number,
        "slot_number": slot_number,
        "result_paths": result_paths,
        "stats_paths": stats_paths,
        "processes": processes,
    }


def finish_self_play_match(
    match,
    run_dir,
    run_id,
    all_replay_data,
    training_archive_interval=TRAINING_ARCHIVE_INTERVAL_MATCHES,
    training_coach_interval=TRAINING_COACH_INTERVAL_MATCHES,
):
    match_no = match["match_number"]
    replay_data = collect_replay_results(match["result_paths"])

    match_replay_path = run_dir / f"match_{match_no:04d}_replays.json"
    match_replay_path.write_text(
        json.dumps(replay_data, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    all_replay_data["matches"].extend(replay_data.get("matches", []))
    all_replay_data["bots"].extend(replay_data.get("bots", []))

    add_general_guess_records(
        collect_general_guess_records(match["result_paths"])
    )

    write_json_file(run_dir / "latest_match.json", replay_data)
    write_json_file(run_dir / "all_replays.json", all_replay_data)

    batch_summary = process_self_play_batch(
        run_dir,
        run_id,
        match_no,
        batch_size=training_archive_interval,
        coach_interval=training_coach_interval,
    )
    return batch_summary


def terminate_self_play_matches(active_matches, reason="force_stopped", dashboard=None):
    for match in active_matches:
        for item in match.get("processes", []):
            process = item.get("process")
            if process is None or not process.is_alive():
                continue

            if dashboard:
                dashboard.set_message(f"Force stopping: {item['label']}.")
            process.terminate()
            process.join(timeout=5)

            if process.is_alive():
                process.kill()
                process.join(timeout=5)

            mark_training_process_failed(item["result_path"], reason)


def run_self_play(
    write_log_files=False,
    parallel_games=1,
    start_delay_seconds=2,
    requeue_delay_seconds=20,
    run_id=None,
    training_archive_interval=TRAINING_ARCHIVE_INTERVAL_MATCHES,
    training_coach_interval=TRAINING_COACH_INTERVAL_MATCHES,
    replay_analyzer_enabled=True,
    replay_analyzer_interval=15,
    use_spawn_grid_hint=False,
    spawn_grid_ab_test=False,
):
    max_parallel_games = len(SELF_PLAY_ACCOUNTS) // 2

    if parallel_games < 1:
        raise ValueError("--parallel-games must be at least 1.")

    if parallel_games > max_parallel_games:
        raise ValueError(
            f"--parallel-games {parallel_games} would need {parallel_games * 2} bots, "
            f"but only {len(SELF_PLAY_ACCOUNTS)} fixed self-play accounts are configured. "
            f"Maximum is {max_parallel_games}."
        )
    if start_delay_seconds < 0:
        raise ValueError("--self-play-start-delay must be zero or greater.")
    if requeue_delay_seconds < 0:
        raise ValueError("--self-play-requeue-delay must be zero or greater.")
    if training_archive_interval < 1:
        raise ValueError("--training-archive-interval must be at least 1.")
    if training_coach_interval < 1:
        raise ValueError("--training-coach-interval must be at least 1.")

    stop_after_current = threading.Event()
    force_stop_now = threading.Event()
    current_runner = {"runner": None}
    runner_lock = threading.Lock()
    match_number = 1
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs/self_play") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "profiles").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

    start_auto_game_listener(
        stop_after_current,
        current_runner,
        runner_lock,
        allow_force_stop=True,
        force_stop_event=force_stop_now,
    )

    all_replay_data = {"matches": [], "bots": []}
    replay_analyzer_status = {}
    dashboard = SelfPlayDashboard(
        run_dir,
        run_id,
        parallel_games,
        start_delay_seconds,
        requeue_delay_seconds,
        replay_analyzer_status,
    )
    dashboard.set_message(
        "Replay analyzer enabled." if replay_analyzer_enabled else "Replay analyzer disabled."
    )
    replay_analyzer = None
    if replay_analyzer_enabled:
        replay_analyzer = BackgroundReplayAnalyzer(
            run_dir,
            run_id=run_id,
            interval_seconds=replay_analyzer_interval,
            max_files_per_pass=4,
            logger=None,
            status=replay_analyzer_status,
        )
        replay_analyzer.start()
    sleep_guard = WindowsSleepGuard("self-play training")

    active_matches = []
    idle_slots = [
        {
            "slot_number": slot_number,
            "available_at": 0.0,
        }
        for slot_number in range(1, parallel_games + 1)
    ]
    last_pair_start_time = 0.0

    try:
        while not stop_after_current.is_set():
            sleep_guard.set_active(bool(active_matches))
            dashboard.render(active_matches, idle_slots, match_number)
            now = time.time()
            if idle_slots and now - last_pair_start_time >= start_delay_seconds:
                idle_slots.sort(key=lambda item: item["available_at"])
                next_slot = idle_slots[0]
                if next_slot["available_at"] <= now:
                    idle_slots.pop(0)
                    dashboard.set_message(
                        f"Dispatching slot {next_slot['slot_number']} to match {match_number:04d}."
                    )
                    active_matches.append(
                        start_self_play_match(
                            match_number=match_number,
                            slot_number=next_slot["slot_number"],
                            run_id=run_id,
                            run_dir=run_dir,
                            write_log_files=write_log_files,
                            use_spawn_grid_hint=use_spawn_grid_hint,
                            spawn_grid_ab_test=spawn_grid_ab_test,
                        )
                    )
                    sleep_guard.set_active(True)
                    match_number += 1
                    last_pair_start_time = time.time()
                    time.sleep(0.1)
                    continue

            still_active_matches = []
            for match in active_matches:
                still_running_processes = []

                for item in match["processes"]:
                    process = item["process"]
                    process.join(timeout=0.1)

                    if not process.is_alive():
                        continue

                    runtime = time.time() - item["started_at"]
                    if runtime <= TRAINING_PROCESS_TIMEOUT_SECONDS:
                        still_running_processes.append(item)
                        continue

                    dashboard.set_message(f"Process timeout: {item['label']}.")
                    process.terminate()
                    process.join(timeout=5)

                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)

                    mark_training_process_timeout(item["result_path"])

                match["processes"] = still_running_processes

                if still_running_processes:
                    still_active_matches.append(match)
                    continue

                batch_summary = finish_self_play_match(
                    match,
                    run_dir,
                    run_id,
                    all_replay_data,
                    training_archive_interval=training_archive_interval,
                    training_coach_interval=training_coach_interval,
                )
                if batch_summary:
                    if batch_summary.get("coach_checkpoint_path"):
                        archive_note = (
                            f" ZIP: {batch_summary['archive_path']}."
                            if batch_summary.get("archive_path")
                            else ""
                        )
                        dashboard.set_message(
                            "Coach checkpoint written at global match "
                            f"{batch_summary['global_finished_matches']}: "
                            f"{batch_summary['coach_checkpoint_path']}."
                            f"{archive_note}"
                        )
                    elif batch_summary.get("archive_path"):
                        dashboard.set_message(
                            "Training ZIP written at global match "
                            f"{batch_summary['archive_threshold']}: "
                            f"{batch_summary['archive_path']}"
                        )
                    else:
                        dashboard.set_message(
                            "Training data updated: "
                            f"+{batch_summary['new_finished_results']} bot results, "
                            f"{batch_summary['global_finished_matches']} global matches."
                        )
                else:
                    dashboard.set_message(f"Finished match {match['match_number']:04d}.")
                available_at = time.time() + requeue_delay_seconds
                idle_slots.append(
                    {
                        "slot_number": match["slot_number"],
                        "available_at": available_at,
                    }
                )
                dashboard.render(still_active_matches, idle_slots, match_number, force=True)

            active_matches = still_active_matches
            sleep_guard.set_active(bool(active_matches))
            time.sleep(1)
    finally:
        if force_stop_now.is_set() and active_matches:
            terminate_self_play_matches(
                active_matches,
                reason="force_stopped",
                dashboard=dashboard,
            )
            active_matches = []
        sleep_guard.allow_sleep()
        if replay_analyzer:
            replay_analyzer.stop()

    dashboard.set_message("Self-play stopped.")
    dashboard.render(active_matches, idle_slots, match_number, force=True)
    print("\nSelf-play stopped.", flush=True)


def run_startup_jax_training():
    if not STARTUP_JAX_TRAINING_ENABLED:
        return None

    print(
        "Startup JAX refresh: "
        f"{STARTUP_JAX_TRAINING_EPOCHS} epochs, "
        f"patience {STARTUP_JAX_TRAINING_PATIENCE}.",
        flush=True,
    )
    with WindowsSleepGuard("startup JAX refresh"):
        policy_result = train_policy_agent(
            epochs=STARTUP_JAX_TRAINING_EPOCHS,
            patience=STARTUP_JAX_TRAINING_PATIENCE,
            continue_from_existing=True,
        )
        action_value_result = train_action_value_agent(
            epochs=STARTUP_JAX_TRAINING_EPOCHS,
            patience=STARTUP_JAX_TRAINING_PATIENCE,
            continue_from_existing=True,
        )

    result = {
        "policy": policy_result,
        "action_value": action_value_result,
    }
    print("Startup JAX refresh:", json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main():
    args = parse_args()
    if args.train_jax_agent:
        with WindowsSleepGuard("JAX policy training"):
            result = train_policy_agent()
        print("JAX policy training:", json.dumps(result, indent=2, sort_keys=True))
        return

    if args.train_action_value_agent:
        with WindowsSleepGuard("JAX action-value training"):
            result = train_action_value_agent()
        print("JAX action-value training:", json.dumps(result, indent=2, sort_keys=True))
        return

    if args.train_spawn_guess_agent:
        with WindowsSleepGuard("JAX spawn guess training"):
            result = train_spawn_guess_agent()
        print("JAX spawn guess training:", json.dumps(result, indent=2, sort_keys=True))
        return

    if not args.skip_startup_training:
        run_startup_jax_training()

    if args.self_play:
        run_self_play(
            write_log_files=args.log_files,
            parallel_games=args.parallel_games,
            start_delay_seconds=args.self_play_start_delay,
            requeue_delay_seconds=args.self_play_requeue_delay,
            run_id=args.self_play_run_id,
            training_archive_interval=args.training_archive_interval,
            training_coach_interval=args.training_coach_interval,
            replay_analyzer_enabled=not args.disable_replay_analyzer,
            replay_analyzer_interval=args.replay_analyzer_interval,
            use_spawn_grid_hint=args.test_spawn_grid_hint,
            spawn_grid_ab_test=args.spawn_grid_ab_test,
        )
        return

    if args.sim_self_play:
        from sim_runner import run_sim_benchmark

        result = run_sim_benchmark(
            games=args.sim_games,
            parallel_games=args.sim_parallel_games,
            grid_size=args.sim_grid_size,
            truncation=args.sim_truncation,
            opponent=args.sim_opponent,
            sim_path=args.sim_path,
        )
        print("Simulator self-play:", json.dumps(result, indent=2, sort_keys=True))
        return

    run_auto_games(
        write_log_files=args.log_files,
        queue_mode=args.queue,
        use_spawn_grid_hint=args.test_spawn_grid_hint,
        requeue=not args.no_requeue,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
