from pathlib import Path

from learning.jax_agent import (
    BIAS_KEYS,
    FEATURE_KEYS,
    MODEL_FILE,
    build_current_sample,
    clamp,
    read_json,
    sample_to_features,
    sigmoid,
)


class JaxPolicyAgent:
    def __init__(self, path=MODEL_FILE):
        self.path = Path(path)
        self.model = read_json(self.path, None)
        if not isinstance(self.model, dict):
            self.model = None

    def is_ready(self):
        return bool(self.model and self.model.get("weights") and self.model.get("feature_keys"))

    def win_probability(self, sample):
        if not self.is_ready():
            return None

        feature_keys = self.model.get("feature_keys") or list(FEATURE_KEYS)
        feature_lookup = dict(zip(FEATURE_KEYS, sample_to_features(sample)))
        features = [feature_lookup.get(key, 0.0) for key in feature_keys]
        weights = self.model.get("weights") or []
        if len(features) != len(weights):
            return None

        logit = float(self.model.get("intercept") or 0.0)
        logit += sum(float(value) * float(weight) for value, weight in zip(features, weights))
        return sigmoid(logit)

    def recommend_for_strategy(self, strategy, coach):
        sample = build_current_sample(strategy, coach)
        probability = self.win_probability(sample)
        if probability is None:
            return None

        if probability >= 0.52:
            return {
                "win_probability": round(probability, 4),
                "bias": {},
                "reason": "model_confident_current_strategy",
            }

        weights_by_feature = dict(zip(self.model.get("feature_keys") or [], self.model.get("weights") or []))
        bias_adjustment = {}
        for key in BIAS_KEYS:
            weight = float(weights_by_feature.get(key, 0.0))
            if abs(weight) < 0.05:
                continue
            bias_adjustment[key] = round(clamp(weight, -0.35, 0.35), 3)

        if not bias_adjustment:
            return None

        return {
            "win_probability": round(probability, 4),
            "bias": bias_adjustment,
            "reason": "model_low_confidence_bias_adjustment",
        }
