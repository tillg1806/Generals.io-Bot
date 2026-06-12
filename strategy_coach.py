import json
from pathlib import Path

from coach_snapshot import CoachSnapshot

PROFILE_FILE = "data/profiles/strategy_profile.json"


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))

class StrategyCoach:
    def __init__(self, path=PROFILE_FILE, logger=None):
        self.path = Path(path)
        self.logger = logger
        self.profile = self.load_profile()
        self.last_snapshot = None
        self.last_mode = None
        self.last_logged_turn = -1
        self.adjustments = {
            "expansion_bias": 0.0,
            "city_bias": 0.0,
            "attack_bias": 0.0,
            "defense_bias": 0.0,
            "route_bias": 0.0,
        }
        self.opponent_bias = {
            "expansion_bias": 0.0,
            "city_bias": 0.0,
            "attack_bias": 0.0,
            "defense_bias": 0.0,
            "route_bias": 0.0,
        }
        self.model_bias = {
            "expansion_bias": 0.0,
            "city_bias": 0.0,
            "attack_bias": 0.0,
            "defense_bias": 0.0,
            "route_bias": 0.0,
        }
        self.game_events = []

    def load_profile(self):
        fallback = {
            "version": 1,
            "games": 0,
            "wins": 0,
            "losses": 0,
            "learned": {
                "expansion_bias": 0.0,
                "city_bias": 0.0,
                "attack_bias": 0.0,
                "defense_bias": 0.0,
                "route_bias": 0.0,
                "reserve_delta": 0,
                "city_focus_delta": 0,
                "general_attack_delta": 0,
            },
            "recent_games": [],
        }
        if not self.path.exists():
            return fallback

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return fallback

        if not isinstance(data, dict):
            return fallback

        learned = data.setdefault("learned", {})
        for key, value in fallback["learned"].items():
            learned.setdefault(key, value)
        data.setdefault("recent_games", [])
        data.setdefault("games", 0)
        data.setdefault("wins", 0)
        data.setdefault("losses", 0)
        data.setdefault("version", 1)
        return data

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.profile, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def apply_start_profile(self, strategy):
        learned = self.profile["learned"]
        strategy.reserve_after_turn = clamp(
            strategy.reserve_after_turn + int(learned.get("reserve_delta", 0)),
            20,
            600,
        )
        strategy.city_focus_after_turn = clamp(
            strategy.city_focus_after_turn + int(learned.get("city_focus_delta", 0)),
            30,
            240,
        )
        strategy.general_attack_after_turn = clamp(
            strategy.general_attack_after_turn + int(learned.get("general_attack_delta", 0)),
            60,
            280,
        )
        strategy.coach_bias = {
            key: float(learned.get(key, 0.0))
            for key in (
                "expansion_bias",
                "city_bias",
                "attack_bias",
                "defense_bias",
                "route_bias",
            )
        }

    def observe(self, state, strategy, turn):
        snapshot = self.build_snapshot(state, turn)
        if snapshot is None:
            return

        self.add_score_deltas(snapshot)
        self.infer_city_pressure(snapshot)
        self.last_snapshot = snapshot
        mode = self.choose_mode(snapshot)
        self.apply_mode(strategy, snapshot, mode)

        if mode != self.last_mode:
            self.game_events.append(
                {
                    "turn": turn,
                    "visible_turn": snapshot.visible_turn,
                    "mode": mode,
                    "my_tiles": snapshot.my_tiles,
                    "enemy_tiles": snapshot.enemy_tiles,
                    "my_army": snapshot.my_army,
                    "enemy_army": snapshot.enemy_army,
                    "visible_my_cities": snapshot.visible_my_cities,
                    "visible_enemy_cities": snapshot.visible_enemy_cities,
                    "suspected_enemy_city_advantage": snapshot.suspected_enemy_city_advantage,
                }
            )
            self.last_mode = mode

        if self.logger and snapshot.visible_turn - self.last_logged_turn >= 25:
            self.last_logged_turn = snapshot.visible_turn
            self.logger(
                "Coach:",
                mode,
                "tiles",
                f"{snapshot.my_tiles}/{snapshot.enemy_tiles}",
                "army",
                f"{snapshot.my_army}/{snapshot.enemy_army}",
                "cities",
                f"{snapshot.visible_my_cities}/{snapshot.visible_enemy_cities}",
                "enemy_city_suspect",
                snapshot.suspected_enemy_city_advantage,
                "bias",
                strategy.coach_bias,
            )

    def build_snapshot(self, state, turn):
        if state.width <= 0 or state.height <= 0 or state.player_index is None:
            return None

        terrain, _ = state.split_map()
        city_indexes = [
            city
            for city in state.city_set()
            if 0 <= city < len(terrain)
        ]
        visible_my_cities = sum(
            1
            for city in city_indexes
            if terrain[city] == state.player_index
        )
        visible_enemy_cities = sum(
            1
            for city in city_indexes
            if terrain[city] >= 0 and terrain[city] != state.player_index
        )

        return CoachSnapshot(
            turn=turn,
            visible_turn=state.visible_turn(turn),
            my_tiles=state.my_tile_count(),
            enemy_tiles=state.biggest_enemy_tile_count(),
            my_army=state.my_total_army(),
            enemy_army=state.biggest_enemy_total_army(),
            visible_enemy_tiles=len(state.visible_enemy_tiles),
            visible_my_cities=visible_my_cities,
            visible_enemy_cities=visible_enemy_cities,
        )

    def add_score_deltas(self, snapshot):
        previous = self.last_snapshot
        if previous is None:
            return

        snapshot.my_army_delta = snapshot.my_army - previous.my_army
        snapshot.enemy_army_delta = snapshot.enemy_army - previous.enemy_army
        snapshot.my_tile_delta = snapshot.my_tiles - previous.my_tiles
        snapshot.enemy_tile_delta = snapshot.enemy_tiles - previous.enemy_tiles

    def infer_city_pressure(self, snapshot):
        if snapshot.visible_enemy_cities > snapshot.visible_my_cities:
            snapshot.suspected_enemy_city_advantage = True
            return

        if self.last_snapshot is None:
            return

        enemy_growth_edge = snapshot.enemy_army_delta - snapshot.my_army_delta
        enemy_not_just_expanding = snapshot.enemy_tile_delta <= snapshot.my_tile_delta + 2
        enough_signal = snapshot.visible_turn >= 35 and enemy_growth_edge >= 6
        high_army_density = (
            snapshot.enemy_tiles > 0
            and snapshot.enemy_army / max(1, snapshot.enemy_tiles)
            > snapshot.my_army / max(1, snapshot.my_tiles) + 0.8
        )

        snapshot.suspected_enemy_city_advantage = (
            enough_signal and enemy_not_just_expanding
        ) or (
            snapshot.visible_turn >= 55 and high_army_density
        )

    def choose_mode(self, snapshot):
        tile_ratio = snapshot.my_tiles / max(1, snapshot.enemy_tiles)
        army_ratio = snapshot.my_army / max(1, snapshot.enemy_army)

        if snapshot.suspected_enemy_city_advantage and army_ratio >= 0.7:
            return "contest_cities"
        if snapshot.visible_enemy_tiles and army_ratio < 0.75:
            return "stabilize"
        if snapshot.visible_turn >= 35 and tile_ratio < 0.75:
            return "catch_up_expand"
        if snapshot.visible_enemy_tiles and army_ratio >= 1.15:
            return "press_attack"
        if snapshot.visible_turn >= 60 and snapshot.my_army > 40 and snapshot.visible_enemy_tiles == 0:
            return "scout_pressure"
        return "balanced"

    def apply_mode(self, strategy, snapshot, mode):
        learned = self.profile["learned"]
        bias = {
            key: float(learned.get(key, 0.0))
            for key in (
                "expansion_bias",
                "city_bias",
                "attack_bias",
                "defense_bias",
                "route_bias",
            )
        }
        for key, value in self.opponent_bias.items():
            bias[key] = bias.get(key, 0.0) + float(value)
        for key, value in self.model_bias.items():
            bias[key] = bias.get(key, 0.0) + float(value)

        if mode == "catch_up_expand":
            bias["expansion_bias"] += 1.2
            bias["route_bias"] += 0.4
            strategy.reserve_after_turn = min(strategy.reserve_after_turn, snapshot.visible_turn + 20)
            strategy.city_focus_after_turn = min(strategy.city_focus_after_turn, snapshot.visible_turn + 20)
        elif mode == "contest_cities":
            bias["city_bias"] += 1.5
            bias["attack_bias"] += 0.5
            bias["route_bias"] += 0.5
            strategy.city_focus_after_turn = min(strategy.city_focus_after_turn, snapshot.visible_turn)
        elif mode == "stabilize":
            bias["defense_bias"] += 1.2
            bias["attack_bias"] -= 0.4
            strategy.reserve_after_turn = max(strategy.reserve_after_turn, snapshot.visible_turn)
        elif mode == "press_attack":
            bias["attack_bias"] += 1.4
            bias["route_bias"] += 0.6
            strategy.general_attack_after_turn = min(strategy.general_attack_after_turn, snapshot.visible_turn)
        elif mode == "scout_pressure":
            bias["expansion_bias"] += 0.6
            bias["route_bias"] += 0.8
            strategy.general_attack_after_turn = min(strategy.general_attack_after_turn, snapshot.visible_turn + 25)
        else:
            bias["route_bias"] += 0.2

        strategy.coach_bias = {
            key: clamp(value, -2.0, 3.0)
            for key, value in bias.items()
        }

    def record_result(self, won, runner):
        if self.last_snapshot is None:
            return

        self.profile["games"] += 1
        if won:
            self.profile["wins"] += 1
        else:
            self.profile["losses"] += 1

        learned = self.profile["learned"]
        tile_ratio = self.last_snapshot.my_tiles / max(1, self.last_snapshot.enemy_tiles)
        army_ratio = self.last_snapshot.my_army / max(1, self.last_snapshot.enemy_army)

        if won:
            self.nudge_learned("attack_bias", 0.08)
            self.nudge_learned("route_bias", 0.04)
        else:
            if self.last_snapshot.suspected_enemy_city_advantage:
                self.nudge_learned("city_bias", 0.18)
                learned["city_focus_delta"] = clamp(
                    int(learned.get("city_focus_delta", 0)) - 5,
                    -80,
                    80,
                )
            if tile_ratio < 0.8:
                self.nudge_learned("expansion_bias", 0.18)
                learned["reserve_delta"] = clamp(int(learned.get("reserve_delta", 0)) + 5, -80, 120)
            if army_ratio < 0.8:
                self.nudge_learned("defense_bias", 0.12)
            if runner.move_count < max(20, self.last_snapshot.visible_turn):
                self.nudge_learned("route_bias", 0.10)
            if self.last_snapshot.visible_enemy_tiles and army_ratio >= 1.0:
                self.nudge_learned("attack_bias", 0.12)
                learned["general_attack_delta"] = clamp(
                    int(learned.get("general_attack_delta", 0)) - 5,
                    -80,
                    80,
                )

        self.profile["recent_games"].insert(
            0,
            {
                "replay_id": runner.replay_id,
                "won": won,
                "move_count": runner.move_count,
                "final_mode": self.last_mode,
                "tile_ratio": round(tile_ratio, 3),
                "army_ratio": round(army_ratio, 3),
                "visible_my_cities": self.last_snapshot.visible_my_cities,
                "visible_enemy_cities": self.last_snapshot.visible_enemy_cities,
                "suspected_enemy_city_advantage": self.last_snapshot.suspected_enemy_city_advantage,
                "events": self.game_events[-12:],
            },
        )
        self.profile["recent_games"] = self.profile["recent_games"][:30]
        self.save()

    def nudge_learned(self, key, amount):
        learned = self.profile["learned"]
        learned[key] = round(clamp(float(learned.get(key, 0.0)) + amount, -1.0, 2.0), 3)
