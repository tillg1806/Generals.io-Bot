import json
import math
from datetime import datetime
from pathlib import Path


POLICY_SAMPLES_FILE = Path("data/training/policy_samples.jsonl")
MODEL_FILE = Path("models/jax_policy_agent.json")

BIAS_KEYS = (
    "expansion_bias",
    "city_bias",
    "attack_bias",
    "defense_bias",
    "route_bias",
)

MODE_KEYS = (
    "balanced",
    "catch_up_expand",
    "contest_cities",
    "press_attack",
    "scout_pressure",
    "stabilize",
    "unknown",
)

FEATURE_KEYS = (
    "reserve_after_turn_norm",
    "city_focus_after_turn_norm",
    "general_attack_after_turn_norm",
    "expansion_bias",
    "city_bias",
    "attack_bias",
    "defense_bias",
    "route_bias",
    "visible_my_cities_norm",
    "visible_enemy_cities_norm",
    "suspected_enemy_city_advantage",
) + tuple(f"mode_{mode}" for mode in MODE_KEYS)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def sigmoid(value):
    value = clamp(value, -60.0, 60.0)
    return 1.0 / (1.0 + math.exp(-value))


def read_json(path, fallback):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def write_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_training_status(model_path, result):
    status_path = Path(model_path).with_suffix(".status.json")
    write_json(
        status_path,
        {
            "version": 1,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "status": result,
        },
    )


def load_jsonl(path):
    records = []
    source = Path(path)
    if not source.exists():
        return records

    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def sample_to_features(sample):
    settings = sample.get("settings") or {}
    coach = sample.get("coach") or {}
    bias = coach.get("bias") or {}
    mode = coach.get("mode") or "unknown"
    if mode not in MODE_KEYS:
        mode = "unknown"

    values = {
        "reserve_after_turn_norm": float(settings.get("reserve_after_turn") or 0) / 600.0,
        "city_focus_after_turn_norm": float(settings.get("city_focus_after_turn") or 0) / 240.0,
        "general_attack_after_turn_norm": float(settings.get("general_attack_after_turn") or 0) / 280.0,
        "visible_my_cities_norm": float(coach.get("visible_my_cities") or 0) / 8.0,
        "visible_enemy_cities_norm": float(coach.get("visible_enemy_cities") or 0) / 8.0,
        "suspected_enemy_city_advantage": 1.0 if coach.get("suspected_enemy_city_advantage") else 0.0,
    }
    for key in BIAS_KEYS:
        values[key] = float(bias.get(key) or 0.0) / 3.0
    for key in MODE_KEYS:
        values[f"mode_{key}"] = 1.0 if key == mode else 0.0

    return [float(values.get(key, 0.0)) for key in FEATURE_KEYS]


def build_current_sample(strategy, coach):
    snapshot = coach.last_snapshot
    return {
        "settings": {
            "reserve_after_turn": getattr(strategy, "reserve_after_turn", 0),
            "city_focus_after_turn": getattr(strategy, "city_focus_after_turn", 0),
            "general_attack_after_turn": getattr(strategy, "general_attack_after_turn", 0),
        },
        "coach": {
            "mode": coach.last_mode or "unknown",
            "bias": getattr(strategy, "coach_bias", {}) or {},
            "visible_my_cities": snapshot.visible_my_cities if snapshot else 0,
            "visible_enemy_cities": snapshot.visible_enemy_cities if snapshot else 0,
            "suspected_enemy_city_advantage": (
                snapshot.suspected_enemy_city_advantage if snapshot else False
            ),
        },
    }


def train_policy_agent(
    samples_path=POLICY_SAMPLES_FILE,
    model_path=MODEL_FILE,
    min_samples=20,
    epochs=350,
    learning_rate=0.08,
):
    samples = [
        sample
        for sample in load_jsonl(samples_path)
        if sample.get("won") is not None
    ]
    if len(samples) < min_samples:
        result = {
            "status": "skipped",
            "reason": "not_enough_samples",
            "sample_count": len(samples),
            "min_samples": min_samples,
        }
        write_training_status(model_path, result)
        return result

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        result = {
            "status": "skipped",
            "reason": "jax_unavailable",
            "error": str(error),
            "sample_count": len(samples),
        }
        write_training_status(model_path, result)
        return result

    features = jnp.array([sample_to_features(sample) for sample in samples], dtype=jnp.float32)
    labels = jnp.array(
        [1.0 if sample.get("won") is True else 0.0 for sample in samples],
        dtype=jnp.float32,
    )

    weights = jnp.zeros((features.shape[1],), dtype=jnp.float32)
    bias = jnp.array(0.0, dtype=jnp.float32)

    def loss_fn(current_weights, current_bias):
        logits = features @ current_weights + current_bias
        prediction = jax.nn.sigmoid(logits)
        eps = 1e-6
        loss = -jnp.mean(
            labels * jnp.log(prediction + eps)
            + (1.0 - labels) * jnp.log(1.0 - prediction + eps)
        )
        return loss + 0.002 * jnp.mean(current_weights * current_weights)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))
    for _ in range(epochs):
        loss, (weight_grad, bias_grad) = grad_fn(weights, bias)
        weights = weights - learning_rate * weight_grad
        bias = bias - learning_rate * bias_grad

    logits = features @ weights + bias
    probabilities = jax.nn.sigmoid(logits)
    predictions = probabilities >= 0.5
    accuracy = float(jnp.mean(predictions == (labels >= 0.5)))

    model = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_keys": list(FEATURE_KEYS),
        "bias_keys": list(BIAS_KEYS),
        "mode_keys": list(MODE_KEYS),
        "weights": [float(value) for value in weights.tolist()],
        "intercept": float(bias),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "loss": round(float(loss), 6),
            "sample_count": len(samples),
            "positive_samples": int(jnp.sum(labels)),
            "negative_samples": int(len(samples) - int(jnp.sum(labels))),
        },
        "status": {
            "status": "trained",
            "sample_count": len(samples),
        },
    }
    write_json(model_path, model)
    return {
        "status": "trained",
        "model_path": str(model_path),
        **model["metrics"],
    }


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
