from pathlib import Path

from learning.jax_agent import (
    ACTION_FEATURE_KEYS,
    ACTION_VALUE_MODEL_FILE,
    read_json,
    sample_to_action_features,
    sigmoid,
)


class JaxActionValueAgent:
    def __init__(self, path=ACTION_VALUE_MODEL_FILE, min_samples_to_use=1000):
        self.path = Path(path)
        self.min_samples_to_use = min_samples_to_use
        self.model = read_json(self.path, None)
        if not isinstance(self.model, dict):
            self.model = None

    def sample_count(self):
        if not self.model:
            return 0
        metrics = self.model.get("metrics") or {}
        status = self.model.get("status") or {}
        return int(metrics.get("sample_count") or status.get("sample_count") or 0)

    def is_ready(self):
        return bool(
            self.model
            and self.model.get("weights")
            and self.model.get("feature_keys")
            and self.sample_count() >= self.min_samples_to_use
        )

    def action_value(self, sample):
        if not self.is_ready():
            return None

        feature_keys = self.model.get("feature_keys") or list(ACTION_FEATURE_KEYS)
        feature_lookup = dict(zip(ACTION_FEATURE_KEYS, sample_to_action_features(sample)))
        features = [feature_lookup.get(key, 0.0) for key in feature_keys]
        weights = self.model.get("weights") or []
        if len(features) != len(weights):
            return None

        logit = float(self.model.get("intercept") or 0.0)
        logit += sum(float(value) * float(weight) for value, weight in zip(features, weights))
        return sigmoid(logit)

    def score_adjustment(self, sample, score_cap=16000):
        probability = self.action_value(sample)
        if probability is None:
            return None

        adjustment = int((probability - 0.5) * 2.0 * score_cap)
        return {
            "action_value": round(probability, 4),
            "score_adjustment": adjustment,
            "sample_count": self.sample_count(),
        }
