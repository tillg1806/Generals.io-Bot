import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from config import TRAINING_ARCHIVE_INTERVAL_MATCHES, TRAINING_COACH_INTERVAL_MATCHES
from learning.jax_agent import train_action_value_agent, train_policy_agent


TRAINING_DIR = Path("data/training")
OPPONENT_PROFILE_FILE = Path("data/opponents/opponent_profiles.json")
ARCHIVE_DIR = Path("archives/self_play")
TRAINING_STATE_FILE = TRAINING_DIR / "training_state.json"
RUNS_DIR = Path("runs/self_play")
PUBLIC_RUNS_DIR = Path("runs/public_1v1")


def read_json(path, fallback):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def write_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path, records):
    records = [record for record in records if record is not None]
    if not records:
        return

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def process_self_play_batch(
    run_dir,
    run_id,
    completed_match_number,
    batch_size=TRAINING_ARCHIVE_INTERVAL_MATCHES,
    coach_interval=TRAINING_COACH_INTERVAL_MATCHES,
):
    return process_self_play_progress(
        run_dir,
        run_id,
        completed_match_number,
        batch_size=batch_size,
        coach_interval=coach_interval,
    )


def process_self_play_progress(
    run_dir,
    run_id,
    completed_match_number,
    batch_size=TRAINING_ARCHIVE_INTERVAL_MATCHES,
    coach_interval=TRAINING_COACH_INTERVAL_MATCHES,
):
    state = load_training_state(batch_size=batch_size, coach_interval=coach_interval)
    new_results = collect_new_finished_results(state)
    if not new_results:
        return None

    grouped_results = group_results_by_source_run(new_results)
    game_summary_count = 0
    policy_sample_count = 0
    action_sample_count = 0
    prediction_sample_count = 0

    for source_run_id, results in grouped_results.items():
        summaries = build_game_summary_records(source_run_id, results)
        policy_samples = build_policy_samples(source_run_id, results)
        action_samples = build_action_samples(source_run_id, results)
        prediction_samples = collect_prediction_samples_for_results(results)
        append_jsonl(TRAINING_DIR / "games_summary.jsonl", summaries)
        append_jsonl(TRAINING_DIR / "policy_samples.jsonl", policy_samples)
        append_jsonl(TRAINING_DIR / "action_samples.jsonl", action_samples)
        append_jsonl(TRAINING_DIR / "prediction_samples.jsonl", prediction_samples)
        game_summary_count += len(summaries)
        policy_sample_count += len(policy_samples)
        action_sample_count += len(action_samples)
        prediction_sample_count += len(prediction_samples)

    update_opponent_profiles(new_results)
    record_processed_results(state, new_results)

    total_matches = int(state.get("total_finished_matches", 0))
    archive_path = None
    archive_threshold = None
    checkpoint_path = None
    checkpoint = None
    if total_matches >= int(state.get("next_archive_match_threshold", batch_size)):
        archive_threshold = int(state.get("next_archive_match_threshold", batch_size))
        archive_path = archive_pending_training_files(state, archive_threshold)
        while total_matches >= int(state.get("next_archive_match_threshold", batch_size)):
            state["next_archive_match_threshold"] = int(state.get("next_archive_match_threshold", batch_size)) + batch_size
    else:
        archive_threshold = None

    while total_matches >= int(state.get("next_coach_match_threshold", coach_interval)):
        coach_threshold = int(state.get("next_coach_match_threshold", coach_interval))
        checkpoint = build_coach_checkpoint(Path(run_dir), run_id, coach_threshold, new_results)
        checkpoint["global_finished_matches"] = total_matches
        checkpoint["jax_policy_training"] = train_policy_agent()
        checkpoint["jax_action_value_training"] = train_action_value_agent()
        checkpoint_path = Path(run_dir) / "coach_checkpoints" / f"global_checkpoint_{coach_threshold:04d}.json"
        write_json(checkpoint_path, checkpoint)
        append_jsonl(TRAINING_DIR / "coach_feedback.jsonl", [checkpoint])
        while total_matches >= int(state.get("next_coach_match_threshold", coach_interval)):
            state["next_coach_match_threshold"] = int(state.get("next_coach_match_threshold", coach_interval)) + coach_interval
        break

    state["last_update_at"] = datetime.now().isoformat(timespec="seconds")
    state["last_run_id"] = run_id
    state["last_completed_match_number"] = completed_match_number
    if archive_path:
        state["last_archive_path"] = str(archive_path)
        state["last_archive_threshold"] = archive_threshold
    if checkpoint_path:
        state["last_coach_checkpoint_path"] = str(checkpoint_path)
    write_json(TRAINING_STATE_FILE, state)

    summary = {
        "run_id": run_id,
        "completed_match_number": completed_match_number,
        "new_finished_results": len(new_results),
        "global_finished_bot_results": state.get("total_finished_bot_results", 0),
        "global_finished_matches": state.get("total_finished_matches", 0),
        "next_archive_match_threshold": state.get("next_archive_match_threshold"),
        "next_coach_match_threshold": state.get("next_coach_match_threshold"),
        "game_summaries_written": game_summary_count,
        "policy_samples_written": policy_sample_count,
        "action_samples_written": action_sample_count,
        "prediction_samples_written": prediction_sample_count,
        "archive_path": str(archive_path) if archive_path else None,
        "archive_threshold": archive_threshold,
        "coach_checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }
    if checkpoint:
        summary["jax_policy_training"] = checkpoint.get("jax_policy_training")
        summary["jax_action_value_training"] = checkpoint.get("jax_action_value_training")

    write_json(Path(run_dir) / "latest_training_progress.json", summary)
    return summary


def process_public_game_result(result_path):
    state = load_training_state()
    path = Path(result_path)
    result = read_json(path, None)
    if not is_finished_training_result(result):
        return {
            "processed": False,
            "reason": "not_finished_training_result",
        }

    key = training_result_key(result, path)
    signature = training_result_signature(result)
    processed = set(state.get("processed_result_keys") or [])
    processed_signatures = set(state.get("processed_result_signatures") or [])
    if key in processed or signature in processed_signatures:
        return {
            "processed": False,
            "reason": "already_processed",
        }

    result["_source_path"] = str(path)
    result["_source_run_path"] = str(path.parent)
    result["_source_run_id"] = path.parent.name
    result["_training_result_key"] = key

    summaries = build_game_summary_records(result["_source_run_id"], [result])
    policy_samples = build_policy_samples(result["_source_run_id"], [result])
    action_samples = build_action_samples(result["_source_run_id"], [result])
    prediction_samples = collect_prediction_samples_for_results([result])

    append_jsonl(TRAINING_DIR / "games_summary.jsonl", summaries)
    append_jsonl(TRAINING_DIR / "policy_samples.jsonl", policy_samples)
    append_jsonl(TRAINING_DIR / "action_samples.jsonl", action_samples)
    append_jsonl(TRAINING_DIR / "prediction_samples.jsonl", prediction_samples)
    update_opponent_profiles([result])
    record_processed_results(state, [result])
    state["last_update_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(TRAINING_STATE_FILE, state)

    return {
        "processed": True,
        "game_summaries_written": len(summaries),
        "policy_samples_written": len(policy_samples),
        "action_samples_written": len(action_samples),
        "prediction_samples_written": len(prediction_samples),
    }


def load_training_state(
    batch_size=TRAINING_ARCHIVE_INTERVAL_MATCHES,
    coach_interval=TRAINING_COACH_INTERVAL_MATCHES,
):
    state = read_json(
        TRAINING_STATE_FILE,
        {
            "version": 1,
            "processed_result_keys": [],
            "processed_result_signatures": existing_training_signatures(),
            "pending_archive_files": [],
            "total_finished_bot_results": 0,
            "total_finished_matches": 0,
            "next_archive_match_threshold": batch_size,
            "next_coach_match_threshold": coach_interval,
        },
    )
    state.setdefault("version", 1)
    state.setdefault("processed_result_keys", [])
    state.setdefault("processed_result_signatures", existing_training_signatures())
    state.setdefault("pending_archive_files", [])
    known_results = max(
        len(state.get("processed_result_keys") or []),
        len(state.get("processed_result_signatures") or []),
    )
    if int(state.get("total_finished_bot_results", 0)) < known_results:
        state["total_finished_bot_results"] = known_results
    state.setdefault("total_finished_bot_results", known_results)
    state["total_finished_matches"] = int(state.get("total_finished_bot_results", known_results)) // 2
    state["next_archive_match_threshold"] = next_training_threshold(
        state.get("next_archive_match_threshold"),
        state["total_finished_matches"],
        batch_size,
    )
    state["next_coach_match_threshold"] = next_training_threshold(
        state.get("next_coach_match_threshold"),
        state["total_finished_matches"],
        coach_interval,
    )
    return state


def next_training_threshold(current_threshold, total_matches, interval):
    interval = max(1, int(interval))
    total_matches = max(0, int(total_matches))
    target_threshold = ((total_matches // interval) + 1) * interval
    try:
        current_threshold = int(current_threshold)
    except (TypeError, ValueError):
        return target_threshold

    if current_threshold <= total_matches or current_threshold > target_threshold:
        return target_threshold
    return current_threshold


def collect_new_finished_results(state):
    processed = set(state.get("processed_result_keys") or [])
    processed_signatures = set(state.get("processed_result_signatures") or [])
    results = []
    for path in sorted(RUNS_DIR.glob("*/*_result.json")):
        result = read_json(path, None)
        if not is_finished_training_result(result):
            continue

        key = training_result_key(result, path)
        signature = training_result_signature(result)
        if key in processed or signature in processed_signatures:
            continue

        result["_source_path"] = str(path)
        result["_source_run_path"] = str(path.parent)
        result["_source_run_id"] = path.parent.name
        result["_training_result_key"] = key
        results.append(result)

    return results


def is_finished_training_result(result):
    return (
        isinstance(result, dict)
        and result.get("status") == "finished"
        and result.get("won") is not None
        and bool(result.get("username"))
        and bool(result.get("replay_id") or result.get("room_id") or result.get("label"))
    )


def training_result_key(result, path):
    replay_id = result.get("replay_id") or "no_replay"
    username = result.get("username") or "unknown_user"
    label = result.get("label") or Path(path).stem
    return f"{replay_id}|{username}|{label}"


def training_result_signature(result):
    replay_id = result.get("replay_id") or result.get("room_id") or "no_replay"
    username = result.get("username") or "unknown_user"
    return f"{replay_id}|{username}"


def existing_training_signatures():
    signatures = []
    path = TRAINING_DIR / "games_summary.jsonl"
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                replay_id = record.get("replay_id") or record.get("room_id")
                username = record.get("username")
                if replay_id and username:
                    signatures.append(f"{replay_id}|{username}")
    except OSError:
        pass
    return sorted(set(signatures))


def group_results_by_source_run(results):
    grouped = defaultdict(list)
    for result in results:
        grouped[result.get("_source_run_id") or "unknown"].append(result)
    return grouped


def record_processed_results(state, results):
    keys = list(state.get("processed_result_keys") or [])
    pending_files = set(state.get("pending_archive_files") or [])
    for result in results:
        key = result.get("_training_result_key")
        if key and key not in keys:
            keys.append(key)
        signature = training_result_signature(result)
        signatures = state.setdefault("processed_result_signatures", [])
        if signature and signature not in signatures:
            signatures.append(signature)
        for path in raw_files_for_result(result):
            pending_files.add(str(path))

    state["processed_result_keys"] = keys
    state["pending_archive_files"] = sorted(pending_files)
    state["processed_result_signatures"] = sorted(set(state.get("processed_result_signatures") or []))
    state["total_finished_bot_results"] = max(len(keys), len(state["processed_result_signatures"]))
    state["total_finished_matches"] = state["total_finished_bot_results"] // 2


def raw_files_for_result(result):
    source_path = Path(result.get("_source_path", ""))
    run_path = Path(result.get("_source_run_path", ""))
    label = result.get("label")
    paths = []
    if source_path.exists():
        paths.append(source_path)
    if run_path.exists() and label:
        match_number = parse_match_number_from_label(label)
        if match_number is not None:
            paths.append(run_path / f"match_{match_number:04d}_replays.json")
        paths.append(run_path / f"{label}_stats.json")
        paths.append(run_path / "predictions" / f"{label}_enemy_predictions.json")
    return [path for path in paths if path.exists() and path.is_file()]


def parse_match_number_from_label(label):
    try:
        if not label or not label.startswith("self-"):
            return None
        return int(label.split("-", 2)[1])
    except (ValueError, IndexError):
        return None


def archive_pending_training_files(state, archive_threshold):
    files = [Path(path) for path in state.get("pending_archive_files") or []]
    files = [path for path in files if path.exists() and path.is_file()]
    if not files:
        state["pending_archive_files"] = []
        return None

    archive_path = ARCHIVE_DIR / "global_batches" / f"self_play_global_{archive_threshold:06d}.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path)

    state["pending_archive_files"] = []
    return archive_path


def collect_batch_files(run_path, batch_start, batch_end, suffix_pattern):
    paths = []
    for match_number in range(batch_start, batch_end + 1):
        paths.extend(sorted(run_path.glob(f"self-{match_number:04d}-*{suffix_pattern.removeprefix('*')}")))
    return paths


def build_batch_summary(run_id, batch_start, batch_end, results):
    finished = [result for result in results if result.get("status") == "finished"]
    wins = sum(1 for result in finished if result.get("won") is True)
    losses = sum(1 for result in finished if result.get("won") is False)
    modes = Counter(result.get("coach_mode") or "unknown" for result in finished)
    variants = Counter(parse_variant_name(result.get("label")) for result in results)
    suspected_city_games = sum(
        1
        for result in finished
        if result.get("coach_suspected_enemy_city_advantage")
    )

    return {
        "run_id": run_id,
        "batch_start": batch_start,
        "batch_end": batch_end,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_files": len(results),
        "finished_results": len(finished),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(finished), 4) if finished else 0,
        "coach_modes": dict(modes),
        "variants": dict(variants),
        "suspected_enemy_city_games": suspected_city_games,
    }


def build_game_summary_records(run_id, results):
    records = []
    opponents_by_room = opponents_by_room_id(results)
    for result in results:
        general_guess = result.get("general_guess") or {}
        final_map = result.get("final_map_analysis") or {}
        record = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "room_id": result.get("room_id"),
            "replay_id": result.get("replay_id"),
            "replay_url": result.get("replay_url"),
            "username": result.get("username"),
            "opponent_username": opponents_by_room.get(result.get("room_id"), {}).get(result.get("username")),
            "label": result.get("label"),
            "variant": parse_variant_name(result.get("label")),
            "status": result.get("status"),
            "won": result.get("won"),
            "reserve_after_turn": result.get("reserve_after_turn"),
            "city_focus_after_turn": result.get("city_focus_after_turn"),
            "general_attack_after_turn": result.get("general_attack_after_turn"),
            "coach_mode": result.get("coach_mode"),
            "coach_bias": result.get("coach_bias"),
            "coach_visible_my_cities": result.get("coach_visible_my_cities"),
            "coach_visible_enemy_cities": result.get("coach_visible_enemy_cities"),
            "coach_suspected_enemy_city_advantage": result.get("coach_suspected_enemy_city_advantage"),
            "guess_correct": general_guess.get("guess_correct"),
            "guess_confidence": general_guess.get("guess_confidence"),
            "width": general_guess.get("width"),
            "height": general_guess.get("height"),
            "symmetry": final_map.get("symmetry"),
        }
        records.append(record)
    return records


def build_policy_samples(run_id, results):
    samples = []
    for result in results:
        if result.get("status") != "finished":
            continue

        samples.append(
            {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "room_id": result.get("room_id"),
                "replay_id": result.get("replay_id"),
                "username": result.get("username"),
                "variant": parse_variant_name(result.get("label")),
                "reward": 1 if result.get("won") is True else -1,
                "won": result.get("won"),
                "settings": {
                    "reserve_after_turn": result.get("reserve_after_turn"),
                    "city_focus_after_turn": result.get("city_focus_after_turn"),
                    "general_attack_after_turn": result.get("general_attack_after_turn"),
                },
                "coach": {
                    "mode": result.get("coach_mode"),
                    "bias": result.get("coach_bias"),
                    "model_bias": result.get("coach_model_bias"),
                    "visible_my_cities": result.get("coach_visible_my_cities"),
                    "visible_enemy_cities": result.get("coach_visible_enemy_cities"),
                    "suspected_enemy_city_advantage": result.get("coach_suspected_enemy_city_advantage"),
                    "events": result.get("coach_events") or [],
                },
                "jax_policy_adjustment": result.get("jax_policy_adjustment"),
                "final_map_analysis": result.get("final_map_analysis"),
            }
        )
    return samples


def build_action_samples(run_id, results):
    samples = []
    for result in results:
        if result.get("status") != "finished":
            continue

        reward = 1 if result.get("won") is True else -1
        for action in result.get("action_samples") or []:
            if not isinstance(action, dict):
                continue
            action_record = {
                key: value
                for key, value in action.items()
                if key not in ("top_candidates",)
            }
            samples.append(
                {
                    **action_record,
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "room_id": result.get("room_id"),
                    "replay_id": result.get("replay_id"),
                    "replay_url": result.get("replay_url"),
                    "username": result.get("username"),
                    "label": result.get("label"),
                    "variant": parse_variant_name(result.get("label")),
                    "reward": reward,
                    "won": result.get("won"),
                    "final_move_count": result.get("move_count"),
                }
            )
    return samples


def collect_prediction_samples(run_path, run_id, batch_start, batch_end):
    samples = []
    for match_number in range(batch_start, batch_end + 1):
        for path in sorted((run_path / "predictions").glob(f"self-{match_number:04d}-*_enemy_predictions.json")):
            data = read_json(path, {})
            for record in data.get("records", []):
                if not isinstance(record, dict):
                    continue
                samples.append(
                    {
                        **record,
                        "run_id": run_id,
                        "source_file": str(path),
                    }
                )
    return samples


def collect_prediction_samples_for_results(results):
    samples = []
    for result in results:
        run_path = Path(result.get("_source_run_path", ""))
        label = result.get("label")
        if not run_path.exists() or not label:
            continue

        path = run_path / "predictions" / f"{label}_enemy_predictions.json"
        data = read_json(path, {})
        for record in data.get("records", []):
            if not isinstance(record, dict):
                continue
            samples.append(
                {
                    **record,
                    "run_id": result.get("_source_run_id"),
                    "source_file": str(path),
                }
            )
    return samples


def update_opponent_profiles(results, path=None):
    if path is None:
        path = OPPONENT_PROFILE_FILE

    data = read_json(path, {"version": 1, "opponents": {}})
    data.setdefault("version", 1)
    opponents = data.setdefault("opponents", {})
    by_room = group_results_by_room(results)

    for room_results in by_room.values():
        if len(room_results) < 2:
            continue

        for result in room_results:
            username = result.get("username")
            opponent = next((item for item in room_results if item.get("username") != username), None)
            if not username or not opponent:
                continue

            opponent_name = opponent.get("username")
            profile = opponents.setdefault(
                opponent_name,
                {
                    "games": 0,
                    "our_wins": 0,
                    "our_losses": 0,
                    "variants": {},
                    "coach_modes": {},
                    "city_pressure_games": 0,
                    "recent_games": [],
                },
            )
            profile["games"] += 1
            if result.get("won") is True:
                profile["our_wins"] += 1
            elif result.get("won") is False:
                profile["our_losses"] += 1

            variant = parse_variant_name(opponent.get("label"))
            profile["variants"][variant] = profile["variants"].get(variant, 0) + 1
            mode = opponent.get("coach_mode") or "unknown"
            profile["coach_modes"][mode] = profile["coach_modes"].get(mode, 0) + 1
            if opponent.get("coach_suspected_enemy_city_advantage"):
                profile["city_pressure_games"] += 1

            profile["last_seen"] = datetime.now().isoformat(timespec="seconds")
            profile["recent_games"].insert(
                0,
                {
                    "room_id": result.get("room_id"),
                    "replay_id": result.get("replay_id"),
                    "our_username": username,
                    "our_variant": parse_variant_name(result.get("label")),
                    "opponent_variant": variant,
                    "our_won": result.get("won"),
                    "opponent_mode": mode,
                    "opponent_bias": opponent.get("coach_bias"),
                },
            )
            profile["recent_games"] = profile["recent_games"][:50]

    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(path, data)


def opponent_adjustment_for_names(names, path=None):
    if path is None:
        path = OPPONENT_PROFILE_FILE

    data = read_json(path, {})
    opponents = data.get("opponents") or {}
    adjustments = []
    for name in names or []:
        profile = opponents.get(name)
        if not profile:
            continue

        games = profile.get("games", 0)
        if games < 3:
            continue

        our_win_rate = profile.get("our_wins", 0) / max(1, games)
        city_pressure_rate = profile.get("city_pressure_games", 0) / max(1, games)
        modes = profile.get("coach_modes") or {}
        dominant_mode = max(modes, key=modes.get) if modes else "unknown"
        bias = {}
        timing = {}
        reasons = []

        if city_pressure_rate >= 0.35:
            bias["city_bias"] = bias.get("city_bias", 0.0) + 0.35
            bias["route_bias"] = bias.get("route_bias", 0.0) + 0.15
            timing["city_focus_delta"] = -10
            reasons.append("opponent often creates city pressure")

        if dominant_mode in ("press_attack", "contest_cities"):
            bias["defense_bias"] = bias.get("defense_bias", 0.0) + 0.25
            timing["reserve_delta"] = 8
            reasons.append(f"opponent often plays {dominant_mode}")

        if our_win_rate < 0.45:
            bias["defense_bias"] = bias.get("defense_bias", 0.0) + 0.2
            bias["route_bias"] = bias.get("route_bias", 0.0) + 0.2
            reasons.append("low historical win rate against this opponent")

        if not reasons:
            continue

        adjustments.append(
            {
                "opponent": name,
                "games": games,
                "our_win_rate": round(our_win_rate, 4),
                "dominant_mode": dominant_mode,
                "city_pressure_rate": round(city_pressure_rate, 4),
                "bias": bias,
                "timing": timing,
                "reasons": reasons,
            }
        )

    return combine_opponent_adjustments(adjustments)


def combine_opponent_adjustments(adjustments):
    if not adjustments:
        return None

    combined_bias = defaultdict(float)
    combined_timing = defaultdict(int)
    reasons = []
    for adjustment in adjustments:
        for key, value in (adjustment.get("bias") or {}).items():
            combined_bias[key] += float(value)
        for key, value in (adjustment.get("timing") or {}).items():
            combined_timing[key] += int(value)
        reasons.extend(adjustment.get("reasons") or [])

    return {
        "opponents": adjustments,
        "bias": {key: round(value, 3) for key, value in combined_bias.items()},
        "timing": dict(combined_timing),
        "reasons": sorted(set(reasons)),
    }


def build_coach_checkpoint(run_path, run_id, completed_match_number, results):
    profile_summaries = {}
    for profile_path in sorted((run_path / "profiles").glob("*.json")):
        profile = read_json(profile_path, {})
        learned = profile.get("learned") or {}
        games = profile.get("games", 0)
        wins = profile.get("wins", 0)
        profile_summaries[profile_path.stem] = {
            "games": games,
            "wins": wins,
            "losses": profile.get("losses", 0),
            "win_rate": round(wins / games, 4) if games else 0,
            "learned": learned,
            "recent_games": profile.get("recent_games", [])[:10],
        }

    return {
        "run_id": run_id,
        "completed_match_number": completed_match_number,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "batch_summary": build_batch_summary(
            run_id,
            max(1, completed_match_number - len(results) // 2 + 1),
            completed_match_number,
            results,
        ),
        "profiles": profile_summaries,
        "recommendations": build_checkpoint_recommendations(profile_summaries),
    }


def build_checkpoint_recommendations(profile_summaries):
    recommendations = []
    for name, summary in profile_summaries.items():
        learned = summary.get("learned") or {}
        if summary.get("games", 0) < 20:
            continue
        if summary.get("win_rate", 0) < 0.45:
            recommendations.append(
                {
                    "profile": name,
                    "action": "review_low_win_rate",
                    "reason": "Profile is underperforming in recent self-play.",
                    "learned": learned,
                }
            )
        if learned.get("city_bias", 0) > 1.2:
            recommendations.append(
                {
                    "profile": name,
                    "action": "keep_city_pressure_awareness",
                    "reason": "Many losses suggest enemy city advantage or city timing mattered.",
                    "learned": learned,
                }
            )
    return recommendations


def archive_self_play_raw_batch(run_path, run_id, batch_start, batch_end):
    files = []
    for match_number in range(batch_start, batch_end + 1):
        files.append(run_path / f"match_{match_number:04d}_replays.json")
        files.extend(sorted(run_path.glob(f"self-{match_number:04d}-*_result.json")))
        files.extend(sorted(run_path.glob(f"self-{match_number:04d}-*_stats.json")))
        files.extend(sorted((run_path / "logs").glob(f"self-{match_number:04d}-*.log")))
        files.extend(sorted((run_path / "predictions").glob(f"self-{match_number:04d}-*_enemy_predictions.json")))

    files = [path for path in files if path.exists() and path.is_file()]
    if not files:
        return None

    archive_path = ARCHIVE_DIR / run_id / f"self_play_{run_id}_{batch_start:04d}_{batch_end:04d}.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(run_path))

    for path in files:
        try:
            path.unlink()
        except OSError:
            pass

    return archive_path


def group_results_by_room(results):
    grouped = defaultdict(list)
    for result in results:
        room_id = result.get("room_id")
        if room_id:
            grouped[room_id].append(result)
    return grouped


def opponents_by_room_id(results):
    lookup = {}
    for room_id, room_results in group_results_by_room(results).items():
        room_lookup = {}
        for result in room_results:
            username = result.get("username")
            opponent = next((item for item in room_results if item.get("username") != username), None)
            if username and opponent:
                room_lookup[username] = opponent.get("username")
        lookup[room_id] = room_lookup
    return lookup


def parse_variant_name(label):
    if not label or "-" not in label:
        return "unknown"
    return label.split("-", 3)[-1] or "unknown"
