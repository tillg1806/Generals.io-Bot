import json
import math
from datetime import datetime
from pathlib import Path


POLICY_SAMPLES_FILE = Path("data/training/policy_samples.jsonl")
ACTION_SAMPLES_FILE = Path("data/training/action_samples.jsonl")
SIM_ACTION_SAMPLES_FILE = Path("data/training/sim_action_samples.jsonl")
MODEL_FILE = Path("models/jax_policy_agent.json")
ACTION_VALUE_MODEL_FILE = Path("models/jax_action_value_agent.json")
SPAWN_GUESS_SAMPLES_FILE = Path("data/replays/duel_map_dataset.jsonl")
SPAWN_GUESS_MODEL_FILE = Path("models/jax_spawn_guess_agent.json")
SPAWN_GRID_MODEL_FILE = Path("models/jax_spawn_grid_agent_v2_early_stop.json")

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

ACTION_FLAG_KEYS = (
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

ACTION_COMPONENT_KEYS = (
    "attack_threat",
    "defend_general",
    "known_general_target",
    "known_general_route",
    "take_city",
    "attack_enemy_tile",
    "new_tile",
    "route_progress",
    "fallback_route_progress",
    "route_gateway",
    "frontier",
    "search_frontier",
    "target_distance",
    "reverse_penalty",
    "edge_repeat_penalty",
    "target_repeat_penalty",
    "lookahead",
    "stalemate_scout",
    "stalemate_new_tile",
    "stalemate_route",
    "stalemate_city",
    "stalemate_repeat_penalty",
    "information_gain",
    "belief_proximity",
    "action_value",
    "option_secure_general",
    "option_attack_general",
    "option_scout_general",
    "option_contest_city",
    "option_break_stalemate",
    "option_press_route",
)

ACTION_FEATURE_KEYS = (
    "visible_turn_norm",
    "move_number_norm",
    "source_army_norm",
    "target_army_norm",
    "target_distance_norm",
    "my_tiles_norm",
    "enemy_tiles_norm",
    "my_army_norm",
    "enemy_army_norm",
    "visible_enemy_tiles_norm",
    "seen_ratio",
    "score_norm",
    "base_score_norm",
    "lookahead_bonus_norm",
) + tuple(f"flag_{key}" for key in ACTION_FLAG_KEYS) + tuple(
    f"component_{key}" for key in ACTION_COMPONENT_KEYS
)

SPAWN_FEATURE_KEYS = (
    "width_norm",
    "height_norm",
    "my_x_norm",
    "my_y_norm",
    "candidate_x_norm",
    "candidate_y_norm",
    "dx_norm",
    "dy_norm",
    "manhattan_norm",
    "rotated_distance_norm",
    "horizontal_distance_norm",
    "vertical_distance_norm",
    "edge_distance_norm",
    "center_distance_norm",
    "blocked_neighbor_ratio",
    "city_neighbor_ratio",
    "candidate_is_city",
    "candidate_is_blocked",
)


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


def load_action_training_samples(samples_path):
    samples = load_jsonl(samples_path)
    if Path(samples_path) == ACTION_SAMPLES_FILE:
        samples.extend(load_jsonl(SIM_ACTION_SAMPLES_FILE))
    return samples


def initial_linear_params(jnp, feature_keys, model_path, continue_from_existing=False):
    feature_count = len(feature_keys)
    if continue_from_existing:
        model = read_json(model_path, None)
        if (
            isinstance(model, dict)
            and model.get("feature_keys") == list(feature_keys)
            and len(model.get("weights") or []) == feature_count
        ):
            return (
                jnp.array(model["weights"], dtype=jnp.float32),
                jnp.array(float(model.get("intercept") or 0.0), dtype=jnp.float32),
                True,
            )

    return (
        jnp.zeros((feature_count,), dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        False,
    )


def should_stop_early(current_loss, best_loss, stale_epochs, patience, min_delta):
    if patience is None or patience <= 0:
        return current_loss, 0, False, True
    if current_loss < best_loss - min_delta:
        return current_loss, 0, False, True
    stale_epochs += 1
    return best_loss, stale_epochs, stale_epochs >= patience, False


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


def sample_to_action_features(sample):
    width = float(sample.get("width") or 0)
    height = float(sample.get("height") or 0)
    tile_count = max(1.0, width * height)
    flags = sample.get("flags") or {}
    components = sample.get("score_components") or {}
    values = {
        "visible_turn_norm": float(sample.get("visible_turn") or 0) / 500.0,
        "move_number_norm": float(sample.get("move_number") or 0) / 1000.0,
        "source_army_norm": float(sample.get("source_army") or 0) / 250.0,
        "target_army_norm": float(sample.get("target_army") or 0) / 250.0,
        "target_distance_norm": float(sample.get("target_distance") or 0) / 60.0,
        "my_tiles_norm": float(sample.get("my_tiles") or 0) / tile_count,
        "enemy_tiles_norm": float(sample.get("enemy_tiles") or 0) / tile_count,
        "my_army_norm": float(sample.get("my_army") or 0) / 1000.0,
        "enemy_army_norm": float(sample.get("enemy_army") or 0) / 1000.0,
        "visible_enemy_tiles_norm": float(sample.get("visible_enemy_tiles") or 0) / tile_count,
        "seen_ratio": float(sample.get("seen_tiles") or 0) / tile_count,
        "score_norm": float(sample.get("score") or 0) / 300000.0,
        "base_score_norm": float(sample.get("base_score") or 0) / 300000.0,
        "lookahead_bonus_norm": float(sample.get("lookahead_bonus") or 0) / 20000.0,
    }
    for key in ACTION_FLAG_KEYS:
        values[f"flag_{key}"] = 1.0 if flags.get(key) else 0.0
    for key in ACTION_COMPONENT_KEYS:
        values[f"component_{key}"] = float(components.get(key) or 0) / 300000.0

    return [float(values.get(key, 0.0)) for key in ACTION_FEATURE_KEYS]


def spawn_candidate_features(width, height, my_general, candidate, city_set=None, blocked_set=None):
    city_set = city_set or set()
    blocked_set = blocked_set or set()
    tile_count = max(1, width * height)
    scale = max(1.0, float(width + height))
    my_x = my_general % width
    my_y = my_general // width
    x = candidate % width
    y = candidate // width
    rotated = (height - 1 - my_y) * width + (width - 1 - my_x)
    horizontal = my_y * width + (width - 1 - my_x)
    vertical = (height - 1 - my_y) * width + my_x

    neighbors = []
    if y > 0:
        neighbors.append(candidate - width)
    if x < width - 1:
        neighbors.append(candidate + 1)
    if y < height - 1:
        neighbors.append(candidate + width)
    if x > 0:
        neighbors.append(candidate - 1)
    neighbor_count = max(1, len(neighbors))
    blocked_neighbors = sum(1 for index in neighbors if index in blocked_set)
    city_neighbors = sum(1 for index in neighbors if index in city_set)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    edge_distance = min(x, width - 1 - x, y, height - 1 - y)

    values = {
        "width_norm": width / 30.0,
        "height_norm": height / 30.0,
        "my_x_norm": my_x / max(1.0, width - 1),
        "my_y_norm": my_y / max(1.0, height - 1),
        "candidate_x_norm": x / max(1.0, width - 1),
        "candidate_y_norm": y / max(1.0, height - 1),
        "dx_norm": abs(x - my_x) / max(1.0, width - 1),
        "dy_norm": abs(y - my_y) / max(1.0, height - 1),
        "manhattan_norm": (abs(x - my_x) + abs(y - my_y)) / scale,
        "rotated_distance_norm": (
            abs((rotated % width) - x) + abs((rotated // width) - y)
        ) / scale,
        "horizontal_distance_norm": (
            abs((horizontal % width) - x) + abs((horizontal // width) - y)
        ) / scale,
        "vertical_distance_norm": (
            abs((vertical % width) - x) + abs((vertical // width) - y)
        ) / scale,
        "edge_distance_norm": edge_distance / max(1.0, min(width, height) / 2.0),
        "center_distance_norm": (abs(x - center_x) + abs(y - center_y)) / scale,
        "blocked_neighbor_ratio": blocked_neighbors / neighbor_count,
        "city_neighbor_ratio": city_neighbors / neighbor_count,
        "candidate_is_city": 1.0 if candidate in city_set else 0.0,
        "candidate_is_blocked": 1.0 if candidate in blocked_set else 0.0,
    }
    return [float(values.get(key, 0.0)) for key in SPAWN_FEATURE_KEYS]


def spawn_features_from_dataset_record(record, candidate):
    return spawn_candidate_features(
        int(record.get("width") or 0),
        int(record.get("height") or 0),
        int((record.get("generals") or [0])[0]),
        int(candidate),
        city_set=set(record.get("cities") or []),
        blocked_set=set(record.get("mountains") or []),
    )


def spawn_features_from_state(state, terrain, candidate):
    city_set = state.city_set()
    blocked_set = {
        index
        for index, value in enumerate(terrain or [])
        if value in (-2, -4)
    }
    return spawn_candidate_features(
        state.width,
        state.height,
        state.my_general_index,
        candidate,
        city_set=city_set,
        blocked_set=blocked_set,
    )


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
    patience=None,
    min_delta=1e-6,
    continue_from_existing=False,
):
    samples = [
        sample
        for sample in load_action_training_samples(samples_path)
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

    weights, bias, initialized_from_model = initial_linear_params(
        jnp,
        FEATURE_KEYS,
        model_path,
        continue_from_existing=continue_from_existing,
    )

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
    best_loss = float("inf")
    best_weights = weights
    best_bias = bias
    stale_epochs = 0
    epochs_run = 0
    stopped_early = False
    loss = loss_fn(weights, bias)
    for _ in range(epochs):
        loss, (weight_grad, bias_grad) = grad_fn(weights, bias)
        weights = weights - learning_rate * weight_grad
        bias = bias - learning_rate * bias_grad
        epochs_run += 1

        current_loss = float(loss_fn(weights, bias))
        best_loss, stale_epochs, stopped_early, improved = should_stop_early(
            current_loss,
            best_loss,
            stale_epochs,
            patience,
            min_delta,
        )
        if improved:
            best_weights = weights
            best_bias = bias
        if stopped_early:
            break

    if patience is not None and patience > 0:
        weights = best_weights
        bias = best_bias
        loss = loss_fn(weights, bias)

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
            "epochs_run": epochs_run,
            "stopped_early": stopped_early,
            "initialized_from_model": initialized_from_model,
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


def train_action_value_agent(
    samples_path=ACTION_SAMPLES_FILE,
    model_path=ACTION_VALUE_MODEL_FILE,
    min_samples=100,
    epochs=300,
    learning_rate=0.06,
    patience=None,
    min_delta=1e-6,
    continue_from_existing=False,
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

    features = jnp.array([sample_to_action_features(sample) for sample in samples], dtype=jnp.float32)
    labels = jnp.array(
        [1.0 if sample.get("won") is True else 0.0 for sample in samples],
        dtype=jnp.float32,
    )
    weights, bias, initialized_from_model = initial_linear_params(
        jnp,
        ACTION_FEATURE_KEYS,
        model_path,
        continue_from_existing=continue_from_existing,
    )

    def loss_fn(current_weights, current_bias):
        logits = features @ current_weights + current_bias
        prediction = jax.nn.sigmoid(logits)
        eps = 1e-6
        loss = -jnp.mean(
            labels * jnp.log(prediction + eps)
            + (1.0 - labels) * jnp.log(1.0 - prediction + eps)
        )
        return loss + 0.003 * jnp.mean(current_weights * current_weights)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))
    best_loss = float("inf")
    best_weights = weights
    best_bias = bias
    stale_epochs = 0
    epochs_run = 0
    stopped_early = False
    loss = loss_fn(weights, bias)
    for _ in range(epochs):
        loss, (weight_grad, bias_grad) = grad_fn(weights, bias)
        weights = weights - learning_rate * weight_grad
        bias = bias - learning_rate * bias_grad
        epochs_run += 1

        current_loss = float(loss_fn(weights, bias))
        best_loss, stale_epochs, stopped_early, improved = should_stop_early(
            current_loss,
            best_loss,
            stale_epochs,
            patience,
            min_delta,
        )
        if improved:
            best_weights = weights
            best_bias = bias
        if stopped_early:
            break

    if patience is not None and patience > 0:
        weights = best_weights
        bias = best_bias
        loss = loss_fn(weights, bias)

    logits = features @ weights + bias
    probabilities = jax.nn.sigmoid(logits)
    predictions = probabilities >= 0.5
    accuracy = float(jnp.mean(predictions == (labels >= 0.5)))
    positive_samples = int(jnp.sum(labels))

    model = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_keys": list(ACTION_FEATURE_KEYS),
        "flag_keys": list(ACTION_FLAG_KEYS),
        "component_keys": list(ACTION_COMPONENT_KEYS),
        "weights": [float(value) for value in weights.tolist()],
        "intercept": float(bias),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "loss": round(float(loss), 6),
            "sample_count": len(samples),
            "positive_samples": positive_samples,
            "negative_samples": len(samples) - positive_samples,
            "epochs_run": epochs_run,
            "stopped_early": stopped_early,
            "initialized_from_model": initialized_from_model,
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


def replay_id_offset(replay_id, modulo):
    if modulo <= 0:
        return 0
    return sum(ord(char) for char in str(replay_id or "")) % modulo


def spawn_training_samples(records, negatives_per_record=24):
    features = []
    labels = []
    record_count = 0
    for record in records:
        generals = record.get("generals") or []
        width = int(record.get("width") or 0)
        height = int(record.get("height") or 0)
        if len(generals) < 2 or width <= 0 or height <= 0:
            continue
        my_general = int(generals[0])
        enemy_general = int(generals[1])
        cities = set(record.get("cities") or [])
        blocked = set(record.get("mountains") or [])
        tile_count = width * height
        if enemy_general < 0 or enemy_general >= tile_count:
            continue

        candidates = [
            index
            for index in range(tile_count)
            if index != my_general and index not in cities and index not in blocked
        ]
        negatives = [index for index in candidates if index != enemy_general]
        if not negatives:
            continue

        features.append(spawn_candidate_features(width, height, my_general, enemy_general, cities, blocked))
        labels.append(1.0)

        take = min(int(negatives_per_record), len(negatives))
        stride = max(1, len(negatives) // take)
        offset = replay_id_offset(record.get("replay_id"), len(negatives))
        sampled = []
        cursor = offset
        while len(sampled) < take:
            candidate = negatives[cursor % len(negatives)]
            if candidate not in sampled:
                sampled.append(candidate)
            cursor += stride

        for candidate in sampled:
            features.append(spawn_candidate_features(width, height, my_general, candidate, cities, blocked))
            labels.append(0.0)
        record_count += 1

    return features, labels, record_count


def train_spawn_guess_agent(
    samples_path=SPAWN_GUESS_SAMPLES_FILE,
    model_path=SPAWN_GUESS_MODEL_FILE,
    min_records=100,
    negatives_per_record=24,
    epochs=280,
    learning_rate=0.09,
    patience=None,
    min_delta=1e-6,
    continue_from_existing=False,
):
    records = load_jsonl(samples_path)
    features_list, labels_list, record_count = spawn_training_samples(
        records,
        negatives_per_record=negatives_per_record,
    )
    if record_count < min_records:
        result = {
            "status": "skipped",
            "reason": "not_enough_records",
            "record_count": record_count,
            "min_records": min_records,
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
            "record_count": record_count,
        }
        write_training_status(model_path, result)
        return result

    features = jnp.array(features_list, dtype=jnp.float32)
    labels = jnp.array(labels_list, dtype=jnp.float32)
    weights, bias, initialized_from_model = initial_linear_params(
        jnp,
        SPAWN_FEATURE_KEYS,
        model_path,
        continue_from_existing=continue_from_existing,
    )
    positive_weight = jnp.array(float(max(1, negatives_per_record)), dtype=jnp.float32)

    def loss_fn(current_weights, current_bias):
        logits = features @ current_weights + current_bias
        prediction = jax.nn.sigmoid(logits)
        eps = 1e-6
        sample_weights = jnp.where(labels >= 0.5, positive_weight, 1.0)
        loss = -jnp.mean(
            sample_weights
            * (
                labels * jnp.log(prediction + eps)
                + (1.0 - labels) * jnp.log(1.0 - prediction + eps)
            )
        )
        return loss + 0.002 * jnp.mean(current_weights * current_weights)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))
    best_loss = float("inf")
    best_weights = weights
    best_bias = bias
    stale_epochs = 0
    epochs_run = 0
    stopped_early = False
    loss = loss_fn(weights, bias)
    for _ in range(epochs):
        loss, (weight_grad, bias_grad) = grad_fn(weights, bias)
        weights = weights - learning_rate * weight_grad
        bias = bias - learning_rate * bias_grad
        epochs_run += 1

        current_loss = float(loss_fn(weights, bias))
        best_loss, stale_epochs, stopped_early, improved = should_stop_early(
            current_loss,
            best_loss,
            stale_epochs,
            patience,
            min_delta,
        )
        if improved:
            best_weights = weights
            best_bias = bias
        if stopped_early:
            break

    if patience is not None and patience > 0:
        weights = best_weights
        bias = best_bias
        loss = loss_fn(weights, bias)

    logits = features @ weights + bias
    probabilities = jax.nn.sigmoid(logits)
    predictions = probabilities >= 0.5
    accuracy = float(jnp.mean(predictions == (labels >= 0.5)))
    positive_scores = probabilities[labels >= 0.5]
    negative_scores = probabilities[labels < 0.5]

    model = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_keys": list(SPAWN_FEATURE_KEYS),
        "weights": [float(value) for value in weights.tolist()],
        "intercept": float(bias),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "loss": round(float(loss), 6),
            "record_count": record_count,
            "sample_count": len(labels_list),
            "positive_samples": int(jnp.sum(labels)),
            "negative_samples": int(len(labels_list) - int(jnp.sum(labels))),
            "avg_positive_score": round(float(jnp.mean(positive_scores)), 4),
            "avg_negative_score": round(float(jnp.mean(negative_scores)), 4),
            "epochs_run": epochs_run,
            "stopped_early": stopped_early,
            "initialized_from_model": initialized_from_model,
        },
        "status": {
            "status": "trained",
            "sample_count": len(labels_list),
            "record_count": record_count,
            "use_in_general_guesser": False,
            "reason": "experimental spawn model; enable only after ranking evaluation is strong enough",
        },
    }
    write_json(model_path, model)
    return {
        "status": "trained",
        "model_path": str(model_path),
        **model["metrics"],
    }
