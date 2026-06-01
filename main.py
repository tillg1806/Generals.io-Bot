import argparse
import json
import multiprocessing as mp
import threading
import time
from datetime import datetime
from pathlib import Path

import socketio
from socketio import SimpleClient

from config import (
    BOT_ENDPOINT,
    BOT_EVENT_IDLE_TIMEOUT_SECONDS,
    BOT_KEY,
    GENERAL_GUESS_STATS_FILE,
    ROOM_ID,
    TRAINING_PROCESS_TIMEOUT_SECONDS,
    TRAINING_SPECTATOR_ROOM_ID,
    TRAINING_USERNAME_PREFIX,
    TILE_FOG,
    TILE_FOG_OBSTACLE,
    TILE_MOUNTAIN,
    USER_ID,
    USERNAME,
)
from game_state import GameState
from general_guesser import GeneralGuesser
from map_analyzer import analyze_state_map
from pathfinding import build_distance_map, distance_to_target
from stats import ReserveTurnStats
from strategy import Strategy


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
        self.replay_id = None
        self.failure_reason = None
        self.last_event_time = time.time()
        self.enemy_general_guess_index = None
        self.enemy_general_guess_turn = None
        self.enemy_general_guess_confidence = None
        self.enemy_general_guess_reason = None
        self.enemy_general_guess_candidates = []
        self.enemy_general_actual_index = None
        self.guess_result_recorded = False
        self.map_metrics = None
        self.final_map_analysis = None

        self.state = GameState()
        self.general_guesser = GeneralGuesser()
        self.stats = ReserveTurnStats(stats_path) if stats_path else ReserveTurnStats()
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

    def log(self, *parts, console=False):
        message = f"[{self.label}] " + " ".join(str(part) for part in parts)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as log_file:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"{timestamp} {message}\n")
        if console or self.verbose_console:
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

        self.log("Mein General Index:", self.state.my_general_index)
        self.log("Gegner General Index:", self.state.enemy_general_index)
        self.log("Sichtbare Gegner-Tiles:", len(self.state.visible_enemy_tiles))
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

        self.log(
            f"Move: {move.source} -> {move.target} | "
            f"Armee: {armies[move.source]} | "
            f"Target-Terrain: {terrain[move.target]} | "
            f"Ziel: {move.strategy_target} | "
            f"Half: {move.half}"
        )

    def run(self):
        self.start_quit_listener()

        try:
            self.client.connect(BOT_ENDPOINT, transports=["websocket"])
            self.log("Connected", console=True)
            self.log(f"Room: https://bot.generals.io/games/{self.room_id}", console=True)
            self.log("Reserve ab sichtbarem Turn:", self.reserve_after_turn)
            self.log("City-Fokus ab sichtbarem Turn:", self.city_focus_after_turn)
            self.log("General-Angriff ab sichtbarem Turn:", self.general_attack_after_turn)

            self.client.emit("set_username", (self.user_id, self.username, BOT_KEY))
            set_username_response = self.client.receive(timeout=5)
            self.log("SET_USERNAME:", set_username_response)
            if not self.can_continue_after_set_username(set_username_response):
                self.running = False
                return None

            self.client.emit("join_private", (self.room_id, self.user_id, BOT_KEY))
            self.log("JOIN_PRIVATE:", self.client.receive(timeout=5), console=True)
            self.last_event_time = time.time()

            while self.running:
                try:
                    if time.time() - self.last_event_time > BOT_EVENT_IDLE_TIMEOUT_SECONDS:
                        self.failure_reason = "event_idle_timeout"
                        self.log(
                            "Keine Server-Events seit",
                            BOT_EVENT_IDLE_TIMEOUT_SECONDS,
                            "Sekunden; Bot wird beendet.",
                            console=True,
                        )
                        self.running = False
                        return None

                    if not self.game_started and time.time() - self.last_force > 2:
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

                    if event[0] == "game_start":
                        data = event[1] if len(event) > 1 else {}
                        self.game_started = True
                        self.replay_id = data.get("replay_id")
                        self.state.start(data)
                        self.log("GAME START!", console=True)
                        self.log("Mein Player Index:", self.state.player_index)
                        self.log("Replay:", self.replay_url(), console=True)

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
                        self.log("GAME ENDE:", event[0], console=True)
                        self.log("Statistik gespeichert fuer Reserve-Turn:", self.reserve_after_turn)
                        self.running = False
                        return won

                except socketio.exceptions.TimeoutError:
                    continue

        except Exception as e:
            self.log("Fehler:", repr(e), console=True)
            return None

        finally:
            try:
                self.client.disconnect()
            except Exception:
                pass

            self.log("Disconnected", console=True)

    def replay_url(self):
        if not self.replay_id:
            return None

        return f"https://bot.generals.io/replays/{self.replay_id}"

    def update_enemy_general_guess(self, turn):
        if self.enemy_general_guess_index is not None:
            terrain, _ = self.state.split_map()
            if self.general_guesser.can_be_general(
                self.state,
                terrain,
                self.enemy_general_guess_index,
            ):
                return
            self.enemy_general_guess_index = None
            self.enemy_general_guess_confidence = None
            self.enemy_general_guess_reason = None
            self.enemy_general_guess_candidates = []
            self.strategy.initial_enemy_general_guess = None
        if self.state.width <= 0 or self.state.height <= 0:
            return
        if self.state.my_general_index is None:
            return

        terrain, _ = self.state.split_map()
        guess = self.general_guesser.choose(self.state, terrain, turn)
        if guess is None:
            return

        self.enemy_general_guess_index = guess.index
        self.enemy_general_guess_turn = turn
        self.enemy_general_guess_confidence = guess.confidence
        self.enemy_general_guess_reason = guess.reason
        self.enemy_general_guess_candidates = guess.candidates
        self.strategy.initial_enemy_general_guess = self.enemy_general_guess_index
        self.log(
            "Gegner-General-Guess:",
            self.enemy_general_guess_index,
            "Confidence:",
            self.enemy_general_guess_confidence,
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
            "guessed_enemy_general_index": self.enemy_general_guess_index,
            "guess_turn": self.enemy_general_guess_turn,
            "height": self.state.height,
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
                "Finale Map-Analyse:",
                "sichtbar =",
                visibility.get("visible_ratio"),
                "vollstaendig =",
                visibility.get("is_full_map_visible"),
                "Symmetrie =",
                self.final_map_analysis.get("symmetry"),
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
        elif "Username too long" in message:
            self.failure_reason = "username_too_long"
        elif "username is taken" in message:
            self.failure_reason = "username_taken"
        else:
            self.failure_reason = "set_username_failed"

        self.log("Username konnte nicht gesetzt werden; Bot wird nicht gestartet.")
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
                "log_path": str(runner.log_path) if runner.log_path else None,
                "general_guess": runner.general_guess_record(won),
                "final_map_analysis": runner.final_map_analysis,
                "won": won,
                "reserve_after_turn": runner.reserve_after_turn,
                "city_focus_after_turn": runner.city_focus_after_turn,
                "general_attack_after_turn": runner.general_attack_after_turn,
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


def run_training_bot(room_id, user_id, username, stats_path, result_path, label, log_path=None):
    runner = BotRunner(
        room_id=room_id,
        user_id=user_id,
        username=username,
        stats_path=stats_path,
        interactive=False,
        label=label,
        log_path=log_path,
    )
    won = runner.run()
    write_bot_result(result_path, runner, won)


def write_pending_bot_result(result_path, room_id, user_id, username, label, log_path=None):
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "room_id": room_id,
                "user_id": user_id,
                "username": username,
                "label": label,
                "log_path": str(log_path) if log_path else None,
                "general_guess": None,
                "final_map_analysis": None,
                "won": None,
                "reserve_after_turn": None,
                "city_focus_after_turn": None,
                "general_attack_after_turn": None,
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


def mark_training_process_timeout(result_path):
    path = Path(result_path)
    if not path.exists():
        return

    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    if result.get("status") != "running":
        return

    result["failure_reason"] = "process_timeout"
    result["status"] = "failed"
    result["won"] = None
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )


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


def merge_stats_file(target_data, stats_path):
    path = Path(stats_path)
    if not path.exists():
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    target_values = target_data.setdefault("reserve_turns", {})
    for reserve_turn, result in data.get("reserve_turns", {}).items():
        target_entry = target_values.setdefault(reserve_turn, {"wins": 0, "losses": 0})
        target_entry["wins"] += result.get("wins", 0)
        target_entry["losses"] += result.get("losses", 0)


def result_counts_for_win_loss_stats(result_path):
    result = read_json_file(result_path, None)
    if not isinstance(result, dict):
        return False

    return result.get("status") == "finished" and result.get("won") is not None


def merge_finished_stats_files(target_data, stats_paths, result_paths):
    for stats_path, result_path in zip(stats_paths, result_paths):
        if result_counts_for_win_loss_stats(result_path):
            merge_stats_file(target_data, stats_path)

    target_city_values = target_data.setdefault("city_focus_turns", {})
    for city_focus_turn, result in data.get("city_focus_turns", {}).items():
        target_entry = target_city_values.setdefault(city_focus_turn, {"wins": 0, "losses": 0})
        target_entry["wins"] += result.get("wins", 0)
        target_entry["losses"] += result.get("losses", 0)

    target_attack_values = target_data.setdefault("general_attack_turns", {})
    for general_attack_turn, result in data.get("general_attack_turns", {}).items():
        target_entry = target_attack_values.setdefault(general_attack_turn, {"wins": 0, "losses": 0})
        target_entry["wins"] += result.get("wins", 0)
        target_entry["losses"] += result.get("losses", 0)


def run_training(instance_count, write_log_files=False):
    raise RuntimeError(
        "Legacy-Training mit TrainingAccountPool wurde entfernt. "
        "Nutze tillbot_pybot.py und pybot_map_analysis.json."
    )

    if instance_count < 2:
        raise ValueError("-t braucht mindestens 2 Instanzen")
    if instance_count % 2 != 0:
        raise ValueError("-t muss gerade sein, damit 1v1-Paare entstehen")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("training_runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    stats_paths = []
    result_paths = []
    accounts_by_username = {}
    match_count = instance_count // 2
    accounts = TrainingAccountPool().ensure_accounts(
        instance_count,
        USER_ID,
        TRAINING_USERNAME_PREFIX,
    )
    accounts_by_username = {
        account["username"]: account
        for account in accounts
    }

    for match_index in range(match_count):
        if match_index == 0:
            room_id = TRAINING_SPECTATOR_ROOM_ID
        else:
            room_id = f"{ROOM_ID}_train_{match_index}"

        for seat in range(2):
            bot_index = match_index * 2 + seat
            stats_path = run_dir / f"bot_{bot_index:02d}_stats.json"
            result_path = run_dir / f"bot_{bot_index:02d}_result.json"
            log_path = run_dir / "logs" / f"bot_{bot_index:02d}.log" if write_log_files else None
            stats_paths.append(stats_path)
            result_paths.append(result_path)
            label = f"train-{match_index}-{seat}"
            write_pending_bot_result(
                result_path,
                room_id,
                accounts[bot_index]["user_id"],
                accounts[bot_index]["username"],
                label,
                log_path,
            )

            process = mp.Process(
                target=run_training_bot,
                args=(
                    room_id,
                    accounts[bot_index]["user_id"],
                    accounts[bot_index]["username"],
                    str(stats_path),
                    str(result_path),
                    label,
                    str(log_path) if log_path else None,
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

    print(
        f"Training gestartet: {instance_count} Instanzen, "
        f"{match_count} parallele 1v1-Spiele, Run: {run_dir}",
        flush=True,
    )
    print(
        f"Spectator-Raum: https://bot.generals.io/games/{TRAINING_SPECTATOR_ROOM_ID}",
        flush=True,
    )
    for match_index in range(1, match_count):
        print(
            f"Trainingsraum {match_index}: https://bot.generals.io/games/{ROOM_ID}_train_{match_index}",
            flush=True,
        )
    if write_log_files:
        print(f"Detail-Logs: {run_dir / 'logs'}", flush=True)

    while processes:
        still_running = []
        for item in processes:
            process = item["process"]
            process.join(timeout=1)
            if not process.is_alive():
                continue

            runtime = time.time() - item["started_at"]
            if runtime <= TRAINING_PROCESS_TIMEOUT_SECONDS:
                still_running.append(item)
                continue

            print(
                "Trainingsprozess haengt zu lange und wird beendet:",
                item["label"],
                "nach",
                int(runtime),
                "Sekunden",
                flush=True,
            )
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            mark_training_process_timeout(item["result_path"])

        processes = still_running

    block_failed_training_accounts(result_paths, accounts_by_username)
    mark_usable_training_accounts(result_paths, accounts_by_username)

    aggregate = {"reserve_turns": {}}
    merge_finished_stats_files(aggregate, stats_paths, result_paths)

    aggregate_path = run_dir / "aggregate_stats.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    replay_results = collect_replay_results(result_paths)
    replay_path = run_dir / "replays.json"
    replay_path.write_text(
        json.dumps(replay_results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    add_general_guess_records(collect_general_guess_records(result_paths))
    global_replay_path = write_global_distinct_replays(run_dir.parent)

    global_stats = ReserveTurnStats()
    for reserve_turn, result in aggregate.get("reserve_turns", {}).items():
        for _ in range(result.get("wins", 0)):
            global_stats.record_result(int(reserve_turn), True)
        for _ in range(result.get("losses", 0)):
            global_stats.record_result(int(reserve_turn), False)
    for city_focus_turn, result in aggregate.get("city_focus_turns", {}).items():
        for _ in range(result.get("wins", 0)):
            global_stats.record_city_focus_result(int(city_focus_turn), True)
        for _ in range(result.get("losses", 0)):
            global_stats.record_city_focus_result(int(city_focus_turn), False)
    for general_attack_turn, result in aggregate.get("general_attack_turns", {}).items():
        for _ in range(result.get("wins", 0)):
            global_stats.record_general_attack_result(int(general_attack_turn), True)
        for _ in range(result.get("losses", 0)):
            global_stats.record_general_attack_result(int(general_attack_turn), False)

    print("Training fertig.")
    print("Sammeldatei:", aggregate_path)
    print("Replays:", replay_path)
    print("Alle Replay-URLs:", global_replay_path)
    print("Globale Statistik aktualisiert: bot_stats.json")


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


def extract_replay_urls(replay_data):
    urls = []

    if isinstance(replay_data, list):
        for item in replay_data:
            if isinstance(item, str):
                urls.append(item)
            elif isinstance(item, dict) and item.get("replay_url"):
                urls.append(item["replay_url"])
        return urls

    if not isinstance(replay_data, dict):
        return urls

    for match in replay_data.get("matches", []):
        if isinstance(match, dict) and match.get("replay_url"):
            urls.append(match["replay_url"])

    for bot in replay_data.get("bots", []):
        if isinstance(bot, dict) and bot.get("replay_url"):
            urls.append(bot["replay_url"])

    return urls


def write_global_distinct_replays(training_runs_dir):
    replay_urls = []
    seen = set()
    training_runs_path = Path(training_runs_dir)

    for replay_path in sorted(training_runs_path.glob("*/replays.json")):
        try:
            replay_data = json.loads(replay_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        for replay_url in extract_replay_urls(replay_data):
            if not replay_url or replay_url in seen:
                continue
            seen.add(replay_url)
            replay_urls.append(replay_url)

    global_replay_path = training_runs_path / "replays.json"
    global_replay_path.write_text(
        json.dumps(replay_urls, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return global_replay_path


def block_failed_training_accounts(result_paths, accounts_by_username):
    failed_usernames = []

    for result_path in result_paths:
        path = Path(result_path)
        if not path.exists():
            continue

        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        username = result.get("username")
        if username not in accounts_by_username or result.get("won") is not None:
            continue
        if result.get("status") == "running":
            continue

        if result.get("failure_reason") == "account_creation_rate_limited":
            print(
                "Account-Erstellung ist rate-limited; Account bleibt fuer spaeter im Pool:",
                username,
                flush=True,
            )
            continue

        if result.get("failure_reason") in ("username_taken", "username_too_long", "set_username_failed"):
            failed_usernames.append(username)

    if failed_usernames:
        TrainingAccountPool().mark_blocked(
            failed_usernames,
            "set_username_or_join_failed",
        )
        print("Blockierte Trainingsaccounts:", ", ".join(failed_usernames), flush=True)


def mark_usable_training_accounts(result_paths, accounts_by_username):
    usable_usernames = []

    for result_path in result_paths:
        path = Path(result_path)
        if not path.exists():
            continue

        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        username = result.get("username")
        if username in accounts_by_username and result.get("won") is not None:
            usable_usernames.append(username)

    if usable_usernames:
        TrainingAccountPool().mark_usable(usable_usernames)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--train",
        type=int,
        metavar="INSTANCES",
        help="Startet Trainingsinstanzen; 10 bedeutet 5 parallele 1v1-Spiele.",
    )
    parser.add_argument(
        "-l",
        "--log-files",
        action="store_true",
        help="Schreibt detaillierte Bot-Logs in Dateien statt auf die Konsole.",
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


def start_auto_game_listener(stop_after_current, current_runner, runner_lock, allow_force_stop=True):
    def listener():
        while not stop_after_current.is_set():
            try:
                command = input().strip().lower()
            except EOFError:
                break

            if command == "e":
                stop_after_current.set()
                print("Autostart gestoppt: laufendes Spiel wird noch beendet.", flush=True)
                break

            if command == "q":
                stop_after_current.set()
                if not allow_force_stop:
                    print("Autostart gestoppt: laufender Trainingsbatch wird noch beendet.", flush=True)
                    break

                with runner_lock:
                    runner = current_runner.get("runner")
                stop_runner_now(runner)
                print("Sofort-Stopp angefordert.", flush=True)
                break

    threading.Thread(target=listener, daemon=True).start()


def run_auto_games(write_log_files=False):
    stop_after_current = threading.Event()
    current_runner = {"runner": None}
    runner_lock = threading.Lock()
    game_number = 1

    print("Auto-Modus aktiv. Eingabe: e = nach aktuellem Spiel stoppen, q = sofort stoppen.", flush=True)
    if write_log_files:
        print("Detail-Logs: single_game_logs", flush=True)
    start_auto_game_listener(stop_after_current, current_runner, runner_lock)

    while not stop_after_current.is_set():
        log_path = None
        if write_log_files:
            log_path = Path("single_game_logs") / f"spiel_{game_number:03d}.log"

        runner = BotRunner(
            interactive=False,
            label=f"spiel-{game_number}",
            log_path=log_path,
        )
        with runner_lock:
            current_runner["runner"] = runner

        print(f"Starte Spiel {game_number}.", flush=True)
        won = runner.run()
        add_general_guess_records([runner.general_guess_record(won)])

        with runner_lock:
            current_runner["runner"] = None

        if stop_after_current.is_set():
            break

        if won is None:
            print("Spiel wurde nicht normal beendet; starte neues Spiel.", flush=True)
            game_number += 1
            time.sleep(2)
            continue

        game_number += 1
        time.sleep(2)

    print("Keine neuen Spiele werden gestartet.", flush=True)


def training_room_id(match_index):
    if match_index == 0:
        return TRAINING_SPECTATOR_ROOM_ID

    return f"{ROOM_ID}_train_{match_index}"


def start_training_match(
    run_dir,
    match_index,
    generation,
    accounts,
    write_log_files,
    stats_paths,
    result_paths,
):
    room_id = training_room_id(match_index)
    processes = []

    for seat in range(2):
        bot_index = match_index * 2 + seat
        label = f"train-{match_index}-{seat}-g{generation:04d}"
        stats_path = run_dir / f"match_{match_index:02d}_seat_{seat}_game_{generation:04d}_stats.json"
        result_path = run_dir / f"match_{match_index:02d}_seat_{seat}_game_{generation:04d}_result.json"
        log_path = (
            run_dir / "logs" / f"match_{match_index:02d}_seat_{seat}_game_{generation:04d}.log"
            if write_log_files
            else None
        )

        stats_paths.append(stats_path)
        result_paths.append(result_path)
        write_pending_bot_result(
            result_path,
            room_id,
            accounts[bot_index]["user_id"],
            accounts[bot_index]["username"],
            label,
            log_path,
        )

        process = mp.Process(
            target=run_training_bot,
            args=(
                room_id,
                accounts[bot_index]["user_id"],
                accounts[bot_index]["username"],
                str(stats_path),
                str(result_path),
                label,
                str(log_path) if log_path else None,
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

    print(
        f"Match {match_index} gestartet: https://bot.generals.io/games/{room_id}",
        flush=True,
    )
    return processes


def finish_training_run(run_dir, stats_paths, result_paths, accounts_by_username):
    block_failed_training_accounts(result_paths, accounts_by_username)
    mark_usable_training_accounts(result_paths, accounts_by_username)

    aggregate = {"reserve_turns": {}}
    merge_finished_stats_files(aggregate, stats_paths, result_paths)

    aggregate_path = run_dir / "aggregate_stats.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    replay_results = collect_replay_results(result_paths)
    replay_path = run_dir / "replays.json"
    replay_path.write_text(
        json.dumps(replay_results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    add_general_guess_records(collect_general_guess_records(result_paths))
    global_replay_path = write_global_distinct_replays(run_dir.parent)

    global_stats = ReserveTurnStats()
    for reserve_turn, result in aggregate.get("reserve_turns", {}).items():
        for _ in range(result.get("wins", 0)):
            global_stats.record_result(int(reserve_turn), True)
        for _ in range(result.get("losses", 0)):
            global_stats.record_result(int(reserve_turn), False)
    for city_focus_turn, result in aggregate.get("city_focus_turns", {}).items():
        for _ in range(result.get("wins", 0)):
            global_stats.record_city_focus_result(int(city_focus_turn), True)
        for _ in range(result.get("losses", 0)):
            global_stats.record_city_focus_result(int(city_focus_turn), False)
    for general_attack_turn, result in aggregate.get("general_attack_turns", {}).items():
        for _ in range(result.get("wins", 0)):
            global_stats.record_general_attack_result(int(general_attack_turn), True)
        for _ in range(result.get("losses", 0)):
            global_stats.record_general_attack_result(int(general_attack_turn), False)

    print("Training fertig.")
    print("Sammeldatei:", aggregate_path)
    print("Replays:", replay_path)
    print("Alle Replay-URLs:", global_replay_path)
    print("Globale Statistik aktualisiert: bot_stats.json")


def run_continuous_training(instance_count, write_log_files=False):
    raise RuntimeError(
        "Legacy-Training mit TrainingAccountPool wurde entfernt. "
        "Nutze tillbot_pybot.py und pybot_map_analysis.json."
    )

    if instance_count < 2:
        raise ValueError("-t braucht mindestens 2 Instanzen")
    if instance_count % 2 != 0:
        raise ValueError("-t muss gerade sein, damit 1v1-Paare entstehen")

    stop_after_current = threading.Event()
    current_runner = {"runner": None}
    runner_lock = threading.Lock()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("training_runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    match_count = instance_count // 2
    accounts = TrainingAccountPool().ensure_accounts(
        instance_count,
        USER_ID,
        TRAINING_USERNAME_PREFIX,
    )
    accounts_by_username = {
        account["username"]: account
        for account in accounts
    }
    stats_paths = []
    result_paths = []
    generations = {match_index: 1 for match_index in range(match_count)}
    active_matches = {}

    print(
        f"Auto-Training aktiv: {instance_count} Instanzen, "
        f"{match_count} parallele 1v1-Spiele, Run: {run_dir}",
        flush=True,
    )
    print("Eingabe: e = keine neuen Matches starten, laufende noch beenden.", flush=True)
    if write_log_files:
        print(f"Detail-Logs: {run_dir / 'logs'}", flush=True)

    start_auto_game_listener(stop_after_current, current_runner, runner_lock, allow_force_stop=False)

    for match_index in range(match_count):
        active_matches[match_index] = start_training_match(
            run_dir,
            match_index,
            generations[match_index],
            accounts,
            write_log_files,
            stats_paths,
            result_paths,
        )

    while active_matches:
        for match_index, processes in list(active_matches.items()):
            still_running = []
            for item in processes:
                process = item["process"]
                process.join(timeout=0)
                if not process.is_alive():
                    continue

                runtime = time.time() - item["started_at"]
                if runtime <= TRAINING_PROCESS_TIMEOUT_SECONDS:
                    still_running.append(item)
                    continue

                print(
                    "Trainingsprozess haengt zu lange und wird beendet:",
                    item["label"],
                    "nach",
                    int(runtime),
                    "Sekunden",
                    flush=True,
                )
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                mark_training_process_timeout(item["result_path"])

            if still_running:
                active_matches[match_index] = still_running
                continue

            active_matches.pop(match_index)
            if stop_after_current.is_set():
                print(f"Match {match_index} beendet; kein Neustart wegen Stop-Signal.", flush=True)
                continue

            generations[match_index] += 1
            time.sleep(2)
            active_matches[match_index] = start_training_match(
                run_dir,
                match_index,
                generations[match_index],
                accounts,
                write_log_files,
                stats_paths,
                result_paths,
            )

        time.sleep(1)

    finish_training_run(run_dir, stats_paths, result_paths, accounts_by_username)


def run_auto_training(instance_count, write_log_files=False):
    stop_after_current = threading.Event()
    current_runner = {"runner": None}
    runner_lock = threading.Lock()
    batch_number = 1

    print("Auto-Training aktiv. Eingabe: e = nach aktuellem Batch stoppen.", flush=True)
    start_auto_game_listener(stop_after_current, current_runner, runner_lock, allow_force_stop=False)

    while not stop_after_current.is_set():
        print(f"Starte Trainingsbatch {batch_number}.", flush=True)
        run_training(instance_count, write_log_files=write_log_files)
        batch_number += 1

    print("Keine neuen Trainingsspiele werden gestartet.", flush=True)


def main():
    args = parse_args()
    if args.train:
        run_continuous_training(args.train, write_log_files=args.log_files)
        return

    run_auto_games(write_log_files=args.log_files)


if __name__ == "__main__":
    mp.freeze_support()
    main()
