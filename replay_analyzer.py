import json
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

from data_pipeline import parse_variant_name


ANALYSIS_DIR = Path("data/analysis")
REPLAY_METADATA_FILE = ANALYSIS_DIR / "replay_metadata.jsonl"
REPLAY_ANALYSIS_INDEX_FILE = ANALYSIS_DIR / "replay_analysis_index.json"


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
    records = [record for record in records if record]
    if not records:
        return

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def prediction_path_for_result(run_dir, result):
    label = result.get("label")
    if not label:
        return None
    return Path(run_dir) / "predictions" / f"{label}_enemy_predictions.json"


def prediction_summary(path):
    if path is None or not Path(path).exists():
        return None

    data = read_json(path, {})
    summary = data.get("summary")
    if isinstance(summary, dict) and summary:
        return summary

    records = data.get("records") or []
    total = len(records)
    matched = sum(1 for record in records if record.get("matched"))
    return {
        "records": total,
        "matched": matched,
        "accuracy": round(matched / total, 4) if total else 0,
    }


def coach_event_summary(events):
    events = events or []
    modes = Counter(event.get("mode") or "unknown" for event in events)
    city_pressure_turns = sum(
        1
        for event in events
        if event.get("suspected_enemy_city_advantage")
    )
    return {
        "event_count": len(events),
        "mode_counts": dict(modes),
        "city_pressure_events": city_pressure_turns,
        "first_mode": events[0].get("mode") if events else None,
        "last_mode": events[-1].get("mode") if events else None,
    }


def map_summary(final_map_analysis):
    final_map_analysis = final_map_analysis or {}
    visibility = final_map_analysis.get("visibility") or {}
    terrain = final_map_analysis.get("terrain") or {}
    cities = final_map_analysis.get("cities") or {}
    armies = final_map_analysis.get("armies") or {}
    spawn = final_map_analysis.get("spawn") or {}
    distance_between_generals = spawn.get("distance_between_generals") or {}

    return {
        "width": final_map_analysis.get("width"),
        "height": final_map_analysis.get("height"),
        "tile_count": final_map_analysis.get("tile_count"),
        "visible_ratio": visibility.get("visible_ratio"),
        "is_full_map_visible": visibility.get("is_full_map_visible"),
        "mountain_density": terrain.get("mountain_density"),
        "passable_density": terrain.get("passable_density"),
        "city_count": cities.get("city_count"),
        "owned_city_count": cities.get("owned_city_count"),
        "symmetry": final_map_analysis.get("symmetry"),
        "army_total": armies.get("total"),
        "general_distance_path": distance_between_generals.get("path"),
        "general_distance_manhattan": distance_between_generals.get("manhattan"),
    }


def build_replay_metadata_record(run_dir, run_id, result_path):
    result = read_json(result_path, None)
    if not isinstance(result, dict):
        return None
    if result.get("status") != "finished" or result.get("won") is None:
        return None
    if not result.get("replay_id"):
        return None

    general_guess = result.get("general_guess") or {}
    prediction_file = prediction_path_for_result(run_dir, result)
    return {
        "analysis_version": 1,
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "result_path": str(result_path),
        "room_id": result.get("room_id"),
        "replay_id": result.get("replay_id"),
        "replay_url": result.get("replay_url"),
        "username": result.get("username"),
        "opponent_names": result.get("opponent_names") or [],
        "label": result.get("label"),
        "variant": parse_variant_name(result.get("label")),
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
            "events": coach_event_summary(result.get("coach_events")),
        },
        "general_guess": {
            "guess_correct": general_guess.get("guess_correct"),
            "guess_confidence": general_guess.get("guess_confidence"),
            "guess_turn": general_guess.get("guess_turn"),
            "guess_reason": general_guess.get("guess_reason"),
            "belief_count": len(general_guess.get("general_beliefs") or []),
            "top_beliefs": (general_guess.get("general_beliefs") or [])[:3],
        },
        "map": map_summary(result.get("final_map_analysis")),
        "prediction": prediction_summary(prediction_file),
        "jax_policy_adjustment": result.get("jax_policy_adjustment"),
        "action_value_model": result.get("action_value_model"),
        "strategy_option": result.get("strategy_option"),
        "opponent_memory_adjustment": result.get("opponent_memory_adjustment"),
        "stalemate": result.get("stalemate"),
    }


def load_analysis_index(path=REPLAY_ANALYSIS_INDEX_FILE):
    data = read_json(path, {"version": 1, "processed_result_files": {}})
    if not isinstance(data, dict):
        data = {"version": 1, "processed_result_files": {}}
    data.setdefault("version", 1)
    data.setdefault("processed_result_files", {})
    return data


def result_file_key(path):
    target = Path(path)
    try:
        stat = target.stat()
    except OSError:
        return str(target)
    return f"{target}|{int(stat.st_mtime)}|{stat.st_size}"


def analyze_replay_metadata_once(run_dir, run_id=None, max_files=10):
    run_path = Path(run_dir)
    run_id = run_id or run_path.name
    index = load_analysis_index()
    processed = index.setdefault("processed_result_files", {})
    records = []
    analyzed_paths = []

    for result_path in sorted(run_path.glob("*_result.json")):
        key = result_file_key(result_path)
        if key in processed:
            continue

        record = build_replay_metadata_record(run_path, run_id, result_path)
        processed[key] = {
            "path": str(result_path),
            "processed_at": datetime.now().isoformat(timespec="seconds"),
            "wrote_record": bool(record),
        }
        if record:
            records.append(record)
            analyzed_paths.append(str(result_path))
        if len(records) >= max_files:
            break

    append_jsonl(REPLAY_METADATA_FILE, records)
    index["updated_at"] = datetime.now().isoformat(timespec="seconds")
    index["metadata_file"] = str(REPLAY_METADATA_FILE)
    write_json(REPLAY_ANALYSIS_INDEX_FILE, index)
    return {
        "records_written": len(records),
        "analyzed_paths": analyzed_paths,
    }


class BackgroundReplayAnalyzer:
    def __init__(self, run_dir, run_id=None, interval_seconds=15, max_files_per_pass=4, logger=None, status=None):
        self.run_dir = Path(run_dir)
        self.run_id = run_id or self.run_dir.name
        self.interval_seconds = interval_seconds
        self.max_files_per_pass = max_files_per_pass
        self.logger = logger
        self.status = status
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self.run, daemon=True, name="replay-analyzer")
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)

    def run(self):
        while not self.stop_event.is_set():
            try:
                result = analyze_replay_metadata_once(
                    self.run_dir,
                    self.run_id,
                    max_files=self.max_files_per_pass,
                )
                if self.status is not None:
                    self.status["last_records_written"] = result["records_written"]
                    self.status["last_run_at"] = datetime.now().isoformat(timespec="seconds")
                    self.status["total_records_written"] = (
                        self.status.get("total_records_written", 0)
                        + result["records_written"]
                    )
                if result["records_written"] and self.logger:
                    self.logger(
                        "Replay analyzer:",
                        result["records_written"],
                        "metadata records written.",
                    )
            except Exception as error:
                if self.status is not None:
                    self.status["last_error"] = repr(error)
                    self.status["last_run_at"] = datetime.now().isoformat(timespec="seconds")
                if self.logger:
                    self.logger("Replay analyzer error:", repr(error))

            self.stop_event.wait(self.interval_seconds)
