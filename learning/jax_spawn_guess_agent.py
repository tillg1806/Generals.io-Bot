from pathlib import Path

from learning.jax_agent import (
    SPAWN_FEATURE_KEYS,
    SPAWN_GUESS_MODEL_FILE,
    read_json,
    sigmoid,
    spawn_features_from_state,
)


class JaxSpawnGuessAgent:
    def __init__(self, path=SPAWN_GUESS_MODEL_FILE):
        self.path = Path(path)
        self.model = read_json(self.path, None)
        if not isinstance(self.model, dict):
            self.model = None

    def is_ready(self):
        status = self.model.get("status") if isinstance(self.model, dict) else {}
        return bool(
            self.model
            and self.model.get("weights")
            and self.model.get("feature_keys")
            and status.get("use_in_general_guesser") is True
        )

    def is_trained(self):
        return bool(self.model and self.model.get("weights") and self.model.get("feature_keys"))

    def probability(self, state, terrain, candidate):
        if not self.is_ready():
            return None
        if state.my_general_index is None or state.width <= 0 or state.height <= 0:
            return None

        feature_keys = self.model.get("feature_keys") or list(SPAWN_FEATURE_KEYS)
        feature_lookup = dict(zip(SPAWN_FEATURE_KEYS, spawn_features_from_state(state, terrain, candidate)))
        features = [feature_lookup.get(key, 0.0) for key in feature_keys]
        weights = self.model.get("weights") or []
        if len(features) != len(weights):
            return None

        logit = float(self.model.get("intercept") or 0.0)
        logit += sum(float(value) * float(weight) for value, weight in zip(features, weights))
        return sigmoid(logit)

    def score_adjustment(self, state, terrain, candidate, score_cap=55):
        probability = self.probability(state, terrain, candidate)
        if probability is None:
            return None

        return {
            "spawn_probability": round(probability, 4),
            "score_adjustment": int(probability * score_cap),
        }
