import json
import time
from pathlib import Path

from config import TRAINING_ARCHIVE_INTERVAL_MATCHES, TRAINING_COACH_INTERVAL_MATCHES
from data_pipeline import next_training_threshold


def count_lines(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def read_json_file(path, fallback):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


class SelfPlayDashboard:
    def __init__(self, run_dir, run_id, parallel_games, start_delay_seconds, requeue_delay_seconds, analyzer_status):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.parallel_games = parallel_games
        self.start_delay_seconds = start_delay_seconds
        self.requeue_delay_seconds = requeue_delay_seconds
        self.analyzer_status = analyzer_status
        self.last_render = 0.0
        self.last_message = "Starting scheduler."

    def set_message(self, message):
        self.last_message = message

    def render(self, active_matches, idle_slots, next_match_number, force=False):
        now = time.time()
        if not force and now - self.last_render < 1.0:
            return
        self.last_render = now

        active_by_slot = {match["slot_number"]: match for match in active_matches}
        idle_by_slot = {slot["slot_number"]: slot for slot in idle_slots}
        lines = [
            "TillBot Self-play Dashboard",
            "",
            f"Run ID: {self.run_id}",
            f"Run dir: {self.run_dir}",
            f"Controls: e = stop after running matches | q = quit immediately",
            "",
            f"Parallel games: {self.parallel_games}",
            f"Pair start delay: {self.start_delay_seconds}s",
            f"Idle requeue delay: {self.requeue_delay_seconds}s",
            f"Next match number: {next_match_number}",
            "",
            "Slots:",
        ]

        for slot_number in range(1, self.parallel_games + 1):
            if slot_number in active_by_slot:
                match = active_by_slot[slot_number]
                running = sum(1 for item in match["processes"] if item["process"].is_alive())
                elapsed = int(now - min(item["started_at"] for item in match["processes"]))
                lines.append(
                    f"  slot {slot_number}: running match {match['match_number']:04d} "
                    f"({running}/2 bots alive, {elapsed}s)"
                )
                continue

            idle = idle_by_slot.get(slot_number)
            if idle:
                wait = max(0, int(idle["available_at"] - now))
                state = "ready" if wait == 0 else f"waiting {wait}s"
                lines.append(f"  slot {slot_number}: idle ({state})")
                continue

            lines.append(f"  slot {slot_number}: unknown")

        result_files = list(self.run_dir.glob("*_result.json"))
        match_files = list(self.run_dir.glob("match_*_replays.json"))
        training_state = read_json_file("data/training/training_state.json", {})
        global_finished_matches = int(training_state.get("total_finished_matches", 0) or 0)
        next_archive_threshold = next_training_threshold(
            training_state.get("next_archive_match_threshold"),
            global_finished_matches,
            TRAINING_ARCHIVE_INTERVAL_MATCHES,
        )
        next_coach_threshold = next_training_threshold(
            training_state.get("next_coach_match_threshold"),
            global_finished_matches,
            TRAINING_COACH_INTERVAL_MATCHES,
        )
        archive_matches_remaining = max(0, next_archive_threshold - global_finished_matches)
        coach_matches_remaining = max(0, next_coach_threshold - global_finished_matches)
        finished_results = 0
        running_results = 0
        for result_path in result_files:
            result = read_json_file(result_path, {})
            if result.get("status") == "finished":
                finished_results += 1
            elif result.get("status") == "running":
                running_results += 1

        lines.extend(
            [
                "",
                "Collected data:",
                f"  match replay files: {len(match_files)}",
                f"  finished bot results: {finished_results}",
                f"  running bot results: {running_results}",
                f"  game summaries: {count_lines('data/training/games_summary.jsonl')}",
                f"  policy samples: {count_lines('data/training/policy_samples.jsonl')}",
                f"  action samples: {count_lines('data/training/action_samples.jsonl')}",
                f"  prediction samples: {count_lines('data/training/prediction_samples.jsonl')}",
                f"  global finished matches: {global_finished_matches}",
                f"  next ZIP archive: global match {next_archive_threshold} "
                f"(in {archive_matches_remaining}, interval {TRAINING_ARCHIVE_INTERVAL_MATCHES})",
                f"  next coach/JAX training: global match {next_coach_threshold} "
                f"(in {coach_matches_remaining}, interval {TRAINING_COACH_INTERVAL_MATCHES})",
                f"  replay metadata records: {count_lines('data/analysis/replay_metadata.jsonl')}",
                "",
                "Replay analyzer:",
                f"  total this run: {self.analyzer_status.get('total_records_written', 0)}",
                f"  last pass wrote: {self.analyzer_status.get('last_records_written', 0)}",
                f"  last pass at: {self.analyzer_status.get('last_run_at', '-')}",
            ]
        )
        if self.analyzer_status.get("last_error"):
            lines.append(f"  last error: {self.analyzer_status['last_error']}")

        lines.extend(["", f"Status: {self.last_message}", ""])
        print("\033[2J\033[H" + "\n".join(lines), end="", flush=True)
