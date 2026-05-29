import json
import random
from pathlib import Path

from config import (
    CITY_FOCUS_TURN_MAX,
    CITY_FOCUS_TURN_MIN,
    GENERAL_ATTACK_TURN_MAX,
    GENERAL_ATTACK_TURN_MIN,
    RESERVE_TURN_MAX,
    RESERVE_TURN_MIN,
    STATS_FILE,
)


class ReserveTurnStats:
    def __init__(self, path=STATS_FILE):
        self.path = Path(path)
        self.data = self._load()

    def _load(self):
        if not self.path.exists():
            return {"reserve_turns": {}}

        try:
            return self.normalize_data(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return {"reserve_turns": {}}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self.normalize_data(self.data)
        self.path.write_text(
            json.dumps(self.data, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def normalize_data(self, data):
        if not isinstance(data, dict):
            return {"reserve_turns": {}}

        normalized = dict(data)
        for key in ("reserve_turns", "city_focus_turns", "general_attack_turns"):
            normalized[key] = self.normalize_tracked_values(normalized.get(key, {}))

        return normalized

    def normalize_tracked_values(self, values):
        if not isinstance(values, dict):
            return {}

        normalized = {}
        for value, result in values.items():
            normalized_key = self.format_turn_key(value)
            entry = normalized.setdefault(normalized_key, {"wins": 0, "losses": 0})
            if isinstance(result, dict):
                entry["wins"] += result.get("wins", 0)
                entry["losses"] += result.get("losses", 0)

        return normalized

    def format_turn_key(self, value):
        return f"{int(value):03d}"

    def choose_value(self):
        return self.choose_tracked_value("reserve_turns", RESERVE_TURN_MIN, RESERVE_TURN_MAX)

    def choose_city_focus_value(self):
        return self.choose_tracked_value("city_focus_turns", CITY_FOCUS_TURN_MIN, CITY_FOCUS_TURN_MAX)

    def choose_general_attack_value(self):
        return self.choose_tracked_value("general_attack_turns", GENERAL_ATTACK_TURN_MIN, GENERAL_ATTACK_TURN_MAX)

    def choose_tracked_value(self, key, minimum, maximum):
        values = self.data.setdefault(key, {})

        if random.random() < 0.25 or not values:
            return random.randint(minimum, maximum)

        def score(item):
            value, result = item
            games = result.get("wins", 0) + result.get("losses", 0)
            if games == 0:
                return 2.0

            win_rate = result.get("wins", 0) / games
            confidence_bonus = 1 / (games + 1)
            return win_rate + confidence_bonus

        best_value, _ = max(values.items(), key=score)
        return int(best_value)

    def record_result(self, reserve_turn, won):
        self.record_tracked_result("reserve_turns", reserve_turn, won)

    def record_city_focus_result(self, city_focus_turn, won):
        self.record_tracked_result("city_focus_turns", city_focus_turn, won)

    def record_general_attack_result(self, general_attack_turn, won):
        self.record_tracked_result("general_attack_turns", general_attack_turn, won)

    def record_tracked_result(self, key, value, won):
        values = self.data.setdefault(key, {})
        entry = values.setdefault(self.format_turn_key(value), {"wins": 0, "losses": 0})

        if won:
            entry["wins"] += 1
        else:
            entry["losses"] += 1

        self.save()
