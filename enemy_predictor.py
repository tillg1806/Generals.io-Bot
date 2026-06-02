import json
from pathlib import Path

from pathfinding import distance_to_target


PREDICTION_FILE = "data/predictions/enemy_predictions.json"


class EnemyAttackPredictor:
    def __init__(self, path=PREDICTION_FILE, horizon=10, match_radius=2):
        self.path = Path(path)
        self.horizon = horizon
        self.match_radius = match_radius
        self.pending = []
        self.records = self.load_records()

    def load_records(self):
        if not self.path.exists():
            return {"version": 1, "records": [], "summary": {}}

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "records": [], "summary": {}}

        if not isinstance(data, dict):
            return {"version": 1, "records": [], "summary": {}}

        data.setdefault("version", 1)
        data.setdefault("records", [])
        data.setdefault("summary", {})
        return data

    def save(self):
        self.records["summary"] = self.build_summary()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.records, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def observe(self, state, turn, replay_id=None):
        self.evaluate_pending(state, turn, replay_id)
        prediction = self.make_prediction(state, turn, replay_id)
        if prediction is not None:
            self.pending.append(prediction)

    def make_prediction(self, state, turn, replay_id):
        terrain, armies = state.split_map()
        if not terrain or state.player_index is None:
            return None

        candidates = []
        for source, owner in enumerate(terrain):
            if owner < 0 or owner == state.player_index:
                continue
            if armies[source] <= 1:
                continue

            for target in state.neighbor_indexes(source):
                target_owner = terrain[target]
                if target_owner != state.player_index:
                    continue

                heat = state.enemy_movement_heat.get(source, 0) + state.enemy_prediction_heat.get(target, 0)
                score = heat * 3 + armies[source] - armies[target]
                if target == state.my_general_index:
                    score += 40
                if armies[source] <= armies[target] + 1:
                    score -= 20

                candidates.append(
                    {
                        "source": source,
                        "target": target,
                        "expected_army": max(0, armies[source] - 1),
                        "score": score,
                        "heat": heat,
                        "target_army": armies[target],
                    }
                )

        if not candidates:
            return None

        best = max(candidates, key=lambda item: item["score"])
        if best["score"] < 3:
            return None

        return {
            "replay_id": replay_id,
            "predicted_at_turn": turn,
            "deadline_turn": turn + self.horizon,
            "source": best["source"],
            "target": best["target"],
            "expected_army": best["expected_army"],
            "score": best["score"],
            "heat": best["heat"],
            "target_army": best["target_army"],
            "matched": False,
        }

    def evaluate_pending(self, state, turn, replay_id):
        if not self.pending:
            return

        remaining = []
        for prediction in self.pending:
            if prediction.get("matched"):
                continue

            match = self.find_match(state, prediction)
            if match is not None:
                self.records["records"].append(
                    {
                        **prediction,
                        "resolved_at_turn": turn,
                        "matched": True,
                        "actual_source": match["source"],
                        "actual_target": match["target"],
                        "actual_estimated_army": match["estimated_army"],
                        "target_distance": distance_to_target(state, prediction["target"], match["target"]),
                    }
                )
                continue

            if turn > prediction["deadline_turn"]:
                self.records["records"].append(
                    {
                        **prediction,
                        "resolved_at_turn": turn,
                        "matched": False,
                        "actual_source": None,
                        "actual_target": None,
                        "actual_estimated_army": None,
                        "target_distance": None,
                    }
                )
                continue

            remaining.append(prediction)

        self.pending = remaining
        if state.enemy_attack_events:
            self.save()

    def find_match(self, state, prediction):
        for event in state.enemy_attack_events:
            target_distance = distance_to_target(state, prediction["target"], event["target"])
            source_distance = distance_to_target(state, prediction["source"], event["source"])
            if target_distance <= self.match_radius and source_distance <= self.match_radius:
                return event
        return None

    def flush(self, state=None, turn=None):
        if turn is None:
            turn = 10**9

        for prediction in self.pending:
            self.records["records"].append(
                {
                    **prediction,
                    "resolved_at_turn": turn,
                    "matched": False,
                    "actual_source": None,
                    "actual_target": None,
                    "actual_estimated_army": None,
                    "target_distance": None,
                }
            )
        self.pending = []
        self.save()

    def build_summary(self):
        records = self.records.get("records", [])
        total = len(records)
        matched = sum(1 for record in records if record.get("matched"))
        return {
            "records": total,
            "matched": matched,
            "accuracy": round(matched / total, 4) if total else 0,
            "horizon": self.horizon,
            "match_radius": self.match_radius,
        }
