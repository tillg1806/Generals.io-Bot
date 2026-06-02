import json
import sys
from datetime import datetime
from pathlib import Path

PYBOT_DIR = Path(__file__).resolve().parents[1] / "Generals" / "PyBot"
if PYBOT_DIR.exists():
    sys.path.insert(0, str(PYBOT_DIR))

REPO_DIR = Path(__file__).resolve().parent

from ggbot.core import PythonBot
import ggbot.utils

from game_state import GameState
from general_guesser import GeneralGuesser
from map_analyzer import analyze_state_map
from stats import ReserveTurnStats
from strategy import Strategy


class TillBot(PythonBot):
    def __init__(self):
        super().__init__()
        self.state = GameState()
        self.general_guesser = GeneralGuesser()
        self.stats = ReserveTurnStats()
        self.reserve_after_turn = self.stats.choose_value()
        self.city_focus_after_turn = self.stats.choose_city_focus_value()
        self.general_attack_after_turn = self.stats.choose_general_attack_value()
        self.strategy = Strategy(
            self.state,
            self.reserve_after_turn,
            self.city_focus_after_turn,
            self.general_attack_after_turn,
            logger=self.log,
        )
        self.started_replay_id = None
        self.enemy_general_guess_index = None
        self.enemy_general_guess_confidence = None
        self.enemy_general_guess_reason = None
        self.enemy_general_guess_candidates = []
        self.analysis_path = REPO_DIR / "data" / "analysis" / "pybot_map_analysis.json"
        self.replays_path = REPO_DIR / "data" / "replays" / "pybot_replays.json"
        self.move_count = 0
        self.skipped_invalid_moves = 0
        self.latest_map_data = None
        self.map_snapshots = []
        self.final_map_analysis = None

    def log(self, *parts):
        if getattr(self, "__DEBUG__", False):
            print("[TillBot]", *parts, flush=True)

    def do_turn(self) -> None:
        if self.queued_moves > 0:
            return

        self.sync_game_state()
        turn = self.game.tick

        if turn == self.state.last_move_turn or self.state.my_general_index is None:
            return

        self.update_enemy_general_guess()
        move = self.strategy.choose_move(turn)
        if move is None:
            return

        if not self.is_legal_adjacent_move(move.source, move.target):
            self.skipped_invalid_moves += 1
            self.log("Invalid move skipped:", move.source, "->", move.target)
            return

        self.move(move.source, move.target, move.half, caller="TillBot")
        self.state.last_move_turn = turn
        self.move_count += 1
        self.write_analysis()
        self.log("Move:", move.source, "->", move.target, "half:", move.half)

    def is_legal_adjacent_move(self, source, target):
        if source < 0 or target < 0:
            return False
        if source >= self.game.size or target >= self.game.size:
            return False
        if self.game.terrain[source] != self.game.player_index:
            return False
        if self.game.armies[source] <= 1:
            return False

        source_x = source % self.game.width
        target_x = target % self.game.width
        if abs(source - target) == self.game.width:
            return True
        if abs(source - target) == 1 and abs(source_x - target_x) == 1:
            return True
        return False

    def sync_game_state(self):
        if self.started_replay_id != self.game.replay_id:
            self.state = GameState()
            self.strategy.state = self.state
            self.started_replay_id = self.game.replay_id
            self.state.start({"playerIndex": self.game.player_index})
            self.enemy_general_guess_index = None
            self.enemy_general_guess_confidence = None
            self.enemy_general_guess_reason = None
            self.enemy_general_guess_candidates = []
            self.move_count = 0
            self.skipped_invalid_moves = 0
            self.latest_map_data = None
            self.map_snapshots = []
            self.final_map_analysis = None

        armies = list(self.game.armies)
        terrain = list(self.game.terrain)
        self.latest_map_data = [self.game.width, self.game.height] + armies + terrain
        self.state.update(
            {
                "turn": self.game.tick,
                "map": self.latest_map_data,
                "cities": list(self.game.cities),
                "generals": self.generals(),
                "scores": self.scores(),
            }
        )
        self.remember_map_snapshot()
        if self.game.tick % 25 == 0:
            self.write_analysis()

    def generals(self):
        player_count = max(len(getattr(self.game, "usernames", [])), self.game.player_index + 1)
        generals = [-1] * player_count
        generals[self.game.player_index] = self.game.own_general

        enemy_general = getattr(self.game, "enemy_general", -1)
        if enemy_general != -1:
            enemy_index = self.enemy_player_index()
            if enemy_index >= len(generals):
                generals.extend([-1] * (enemy_index - len(generals) + 1))
            generals[enemy_index] = enemy_general

        return generals

    def enemy_player_index(self):
        for index in range(max(len(getattr(self.game, "usernames", [])), 2)):
            if index != self.game.player_index:
                return index
        return 1

    def scores(self):
        own_tiles = list(self.game.own_tiles)
        enemy_tiles = list(self.game.enemy_tiles)
        return [
            {
                "i": self.game.player_index,
                "tiles": len(own_tiles),
                "total": sum(tile.strength for tile in own_tiles),
                "dead": False,
            },
            {
                "i": self.enemy_player_index(),
                "tiles": len(enemy_tiles),
                "total": sum(tile.strength for tile in enemy_tiles),
                "dead": False,
            },
        ]

    def update_enemy_general_guess(self):
        if self.enemy_general_guess_index is not None:
            terrain, _ = self.state.split_map()
            if self.general_guesser.can_be_general(
                self.state,
                terrain,
                self.enemy_general_guess_index,
            ):
                return
            self.enemy_general_guess_index = None
            self.enemy_general_guess_confidence = None
            self.enemy_general_guess_reason = None
            self.enemy_general_guess_candidates = []
            self.strategy.initial_enemy_general_guess = None
        if self.state.width <= 0 or self.state.height <= 0:
            return
        if self.state.my_general_index is None:
            return

        terrain, _ = self.state.split_map()
        guess = self.general_guesser.choose(self.state, terrain, self.game.tick)
        if guess is None:
            return

        self.enemy_general_guess_index = guess.index
        self.enemy_general_guess_confidence = guess.confidence
        self.enemy_general_guess_reason = guess.reason
        self.enemy_general_guess_candidates = guess.candidates
        self.strategy.initial_enemy_general_guess = self.enemy_general_guess_index

    def on_state_message(self, data):
        if "game_won" in data:
            self.sync_final_game_state()
            self.update_final_map_analysis("won")
            self.write_analysis(status="won")
        elif "game_lost" in data:
            self.sync_final_game_state()
            self.update_final_map_analysis("lost")
            self.write_analysis(status="lost")
        elif "game_start" in data:
            self.write_analysis(status="started")

    def sync_final_game_state(self):
        try:
            self.sync_game_state()
        except Exception as exc:
            self.log("Final sync failed:", repr(exc))

    def remember_map_snapshot(self):
        if self.latest_map_data is None:
            return
        if self.game.tick % 25 != 0 and self.game.tick > 1:
            return

        terrain, armies = self.state.split_map()
        self.map_snapshots.append(
            {
                "turn": self.game.tick,
                "visible_enemy_tiles": len(self.state.visible_enemy_tiles),
                "my_tiles": self.state.my_tile_count(),
                "enemy_tiles": self.state.biggest_enemy_tile_count(),
                "my_total_army": self.state.my_total_army(),
                "enemy_total_army": self.state.biggest_enemy_total_army(),
                "owned_edge_tiles": self.owned_edge_tiles(terrain),
                "map_data": self.latest_map_data,
            }
        )
        self.map_snapshots = self.map_snapshots[-80:]

    def owned_edge_tiles(self, terrain):
        if self.state.width <= 0 or self.state.height <= 0:
            return []

        edge_tiles = []
        for index, owner in enumerate(terrain):
            if owner != self.state.player_index:
                continue
            x = index % self.state.width
            y = index // self.state.width
            if x in (0, self.state.width - 1) or y in (0, self.state.height - 1):
                edge_tiles.append(index)
        return edge_tiles

    def update_final_map_analysis(self, status):
        self.final_map_analysis = analyze_state_map(
            self.state,
            self.latest_map_data,
            status=status,
        )

    def write_analysis(self, status="running"):
        if self.latest_map_data is None:
            return

        if status in ("won", "lost") and self.final_map_analysis is None:
            self.update_final_map_analysis(status)

        records = self.read_analysis_records()
        records[self.started_replay_id or "unknown"] = {
            "analysis_version": 2,
            "status": status,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "replay_id": self.started_replay_id,
            "replay_url": (
                f"https://bot.generals.io/replays/{self.started_replay_id}"
                if self.started_replay_id
                else None
            ),
            "turn": getattr(self.game, "tick", None),
            "player_index": self.state.player_index,
            "my_general_index": self.state.my_general_index,
            "enemy_general_index": self.state.enemy_general_index,
            "enemy_general_guess_index": self.enemy_general_guess_index,
            "enemy_general_guess_confidence": self.enemy_general_guess_confidence,
            "enemy_general_guess_reason": self.enemy_general_guess_reason,
            "enemy_general_guess_candidates": self.enemy_general_guess_candidates,
            "width": self.state.width,
            "height": self.state.height,
            "final_map_analysis": self.final_map_analysis,
            "move_count": self.move_count,
            "skipped_invalid_moves": self.skipped_invalid_moves,
            "latest_map_data": self.latest_map_data,
            "snapshots": self.map_snapshots,
        }
        self.analysis_path.parent.mkdir(parents=True, exist_ok=True)
        self.analysis_path.write_text(
            json.dumps(records, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.write_replay_index(status)

    def read_analysis_records(self):
        if not self.analysis_path.exists():
            return {}

        try:
            data = json.loads(self.analysis_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        return data if isinstance(data, dict) else {}

    def write_replay_index(self, status):
        if not self.started_replay_id:
            return

        replays = self.read_replay_index()
        existing = {
            replay.get("replay_id"): replay
            for replay in replays
            if isinstance(replay, dict) and replay.get("replay_id")
        }
        record = existing.get(self.started_replay_id, {})
        record.update(
            {
                "replay_id": self.started_replay_id,
                "replay_url": f"https://bot.generals.io/replays/{self.started_replay_id}",
                "status": status,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "turn": getattr(self.game, "tick", None),
                "player_index": self.state.player_index,
                "game_type": getattr(self.game, "game_type", None),
                "usernames": list(getattr(self.game, "usernames", []) or []),
                "move_count": self.move_count,
                "analysis_file": str(self.analysis_path.name),
            }
        )
        existing[self.started_replay_id] = record

        sorted_replays = sorted(
            existing.values(),
            key=lambda replay: replay.get("updated_at", ""),
            reverse=True,
        )
        self.replays_path.parent.mkdir(parents=True, exist_ok=True)
        self.replays_path.write_text(
            json.dumps(sorted_replays, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def read_replay_index(self):
        if not self.replays_path.exists():
            return []

        try:
            data = json.loads(self.replays_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        return data if isinstance(data, list) else []


if __name__ == "__main__":
    config = ggbot.utils.get_config_from_cmdline_args()
    TillBot().with_config(config).run()
