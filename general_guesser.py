from pathlib import Path
import json
import math

from config import GENERAL_GUESS_STATS_FILE, SPAWN_MIN_DISTANCE_1V1, TILE_FOG_OBSTACLE, TILE_MOUNTAIN
from general_guess import GeneralGuess
from learning.jax_spawn_grid_agent import JaxSpawnGridAgent
from learning.jax_spawn_guess_agent import JaxSpawnGuessAgent
from pathfinding import build_distance_map, distance_to_center, distance_to_target, is_passable, xy_to_index


class GeneralGuesser:
    def __init__(self, stats_path=GENERAL_GUESS_STATS_FILE, force_spawn_grid=False):
        self.stats_path = Path(stats_path)
        self.stats = self.load_stats()
        self.spawn_guess_agent = JaxSpawnGuessAgent()
        self.spawn_grid_agent = JaxSpawnGridAgent()
        self.force_spawn_grid = force_spawn_grid

        if self.force_spawn_grid and self.spawn_grid_agent.is_trained():
            self.spawn_grid_agent.force_enabled = True

    def load_stats(self):
        if not self.stats_path.exists():
            return {}

        try:
            data = json.loads(self.stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        return data if isinstance(data, dict) else {}

    def choose(self, state, terrain=None, turn=None, top_n=5):
        if state.my_general_index is None or state.width <= 0 or state.height <= 0:
            return None

        if terrain is None:
            terrain, _ = state.split_map()

        tile_count = state.width * state.height
        if len(terrain) < tile_count:
            return None

        candidates = []
        distances_from_general = None
        spawn_grid_scores = self.spawn_grid_agent.score_map(state, terrain)

        for index in range(tile_count):
            if not self.can_be_general(state, terrain, index):
                continue

            if distances_from_general is None:
                distances_from_general = build_distance_map(
                    state,
                    state.my_general_index,
                    terrain,
                    avoid_cities=False,
                )

            score, reasons = self.score_candidate(
                state,
                terrain,
                index,
                distances_from_general,
                turn,
                spawn_grid_scores,
            )
            candidates.append(
                {
                    "index": index,
                    "score": round(score, 2),
                    "reason": ", ".join(reasons[:4]),
                }
            )

        if not candidates:
            return None

        candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
        ranked_candidates = self.with_beliefs(candidates[:top_n])
        best = candidates[0]
        runner_up_score = candidates[1]["score"] if len(candidates) > 1 else best["score"] - 20
        confidence = self.confidence(best["score"], runner_up_score)

        return GeneralGuess(
            index=best["index"],
            score=best["score"],
            confidence=confidence,
            reason=best["reason"],
            candidates=ranked_candidates,
        )

    def with_beliefs(self, candidates, temperature=18.0):
        if not candidates:
            return []

        best_score = max(candidate["score"] for candidate in candidates)
        weights = [
            math.exp((candidate["score"] - best_score) / temperature)
            for candidate in candidates
        ]
        total = sum(weights) or 1.0
        ranked = []
        for rank, (candidate, weight) in enumerate(zip(candidates, weights), start=1):
            ranked.append(
                {
                    **candidate,
                    "rank": rank,
                    "belief": round(weight / total, 4),
                }
            )
        return ranked

    def can_be_general(self, state, terrain, index):
        if index == state.my_general_index:
            return False
        if terrain[index] == state.player_index:
            return False
        if terrain[index] == TILE_MOUNTAIN:
            return False
        if terrain[index] == TILE_FOG_OBSTACLE:
            return False
        if state.has_seen(index):
            return False
        if index in state.city_set():
            return False
        if self.uses_duel_spawn_distance_rule(state):
            if distance_to_target(state, state.my_general_index, index) < SPAWN_MIN_DISTANCE_1V1:
                return False
        return True

    def uses_duel_spawn_distance_rule(self, state):
        if state.my_general_index is None:
            return False

        active_scores = [
            score
            for score in state.scores or []
            if not score.get("dead", False)
        ]
        if 0 < len(active_scores) <= 2:
            return True

        # Public 1v1 maps are small; FFA maps are much larger. Use this as a
        # fallback before scores are populated.
        return max(state.width, state.height) < 30

    def score_candidate(self, state, terrain, index, distances_from_general, turn, spawn_grid_scores=None):
        score = 0.0
        reasons = []

        transform_score, transform_reason = self.spawn_transform_score(state, index)
        score += transform_score
        if transform_reason:
            reasons.append(transform_reason)

        distance_score, distance_reason = self.distance_score(state, index, distances_from_general)
        score += distance_score
        if distance_reason:
            reasons.append(distance_reason)

        visibility_score, visibility_reason = self.visibility_score(state, terrain, index, turn)
        score += visibility_score
        if visibility_reason:
            reasons.append(visibility_reason)

        shape_score, shape_reason = self.map_shape_score(state, terrain, index)
        score += shape_score
        if shape_reason:
            reasons.append(shape_reason)

        model_score, model_reason = self.spawn_model_score(state, terrain, index, spawn_grid_scores)
        score += model_score
        if model_reason:
            reasons.append(model_reason)

        heat_score, heat_reason = self.enemy_heat_score(state, index)
        score += heat_score
        if heat_reason:
            reasons.append(heat_reason)

        return score, reasons

    def spawn_transform_score(self, state, index):
        my_x = state.my_general_index % state.width
        my_y = state.my_general_index // state.width
        x = index % state.width
        y = index // state.width

        transforms = [
            (state.width - 1 - my_x, state.height - 1 - my_y, 70, "rotated spawn"),
            (state.width - 1 - my_x, my_y, 34, "horizontal mirror"),
            (my_x, state.height - 1 - my_y, 34, "vertical mirror"),
            (0, 0, 12, "corner fallback"),
            (state.width - 1, 0, 12, "corner fallback"),
            (0, state.height - 1, 12, "corner fallback"),
            (state.width - 1, state.height - 1, 12, "corner fallback"),
        ]

        best_score = 0
        best_reason = None
        for target_x, target_y, weight, reason in transforms:
            target = xy_to_index(state, target_x, target_y)
            distance = distance_to_target(state, index, target)
            candidate_score = max(0, weight - distance * 9)
            if candidate_score > best_score:
                best_score = candidate_score
                best_reason = reason

        return best_score, best_reason

    def distance_score(self, state, index, distances_from_general):
        path_distance = distances_from_general.get(index)
        manhattan = distance_to_target(state, state.my_general_index, index)
        distance = path_distance if path_distance is not None else manhattan
        expected = self.expected_general_distance(state)
        spread = max(4, (state.width + state.height) * 0.18)
        score = max(0, 30 - abs(distance - expected) * (30 / spread))
        return score, f"distance {distance:g} near expected {expected:g}"

    def expected_general_distance(self, state):
        summary = self.stats.get("summary", {})
        by_map_size = summary.get("by_map_size", {})
        map_entry = by_map_size.get(f"{state.width}x{state.height}", {})
        expected = map_entry.get("avg_actual_path_distance")
        if expected:
            return expected

        return max(8, (state.width + state.height) * 0.55)

    def visibility_score(self, state, terrain, index, turn):
        if not state.has_seen(index):
            return 22, "unseen tile"

        if turn is None or turn < 0:
            return -4, "already scouted"

        visible_turn = state.visible_turn(turn)
        last_seen = state.visible_turn(state.last_seen(index))
        if last_seen >= 0 and visible_turn - last_seen > 35:
            return 8, "stale scout"

        if terrain[index] >= 0:
            return -18, "visible owned tile"

        return -4, "recently scouted"

    def map_shape_score(self, state, terrain, index):
        center_penalty = distance_to_center(state, index)
        edge_distance = min(
            index % state.width,
            state.width - 1 - (index % state.width),
            index // state.width,
            state.height - 1 - (index // state.width),
        )
        score = min(18, center_penalty * 1.5) - max(0, 3 - edge_distance) * 3

        blocked_neighbors = 0
        for neighbor in state.neighbor_indexes(index):
            if not is_passable(terrain[neighbor]):
                blocked_neighbors += 1

        score -= blocked_neighbors * 2
        return score, "far from center"

    def enemy_heat_score(self, state, index):
        movement_heat = state.enemy_movement_heat.get(index, 0)
        prediction_heat = state.enemy_prediction_heat.get(index, 0)
        heat = movement_heat + prediction_heat
        if heat <= 0:
            return 0, None

        return min(25, heat * 3), "enemy movement heat"

    def spawn_model_score(self, state, terrain, index, spawn_grid_scores=None):
        grid_score = (spawn_grid_scores or {}).get(index)
        if grid_score:
            score = grid_score["score_adjustment"]
            probability = grid_score["spawn_grid_probability"]
            return score, f"duel grid model {probability:.3f}"

        if not self.spawn_guess_agent.is_ready():
            return 0, None

        result = self.spawn_guess_agent.score_adjustment(state, terrain, index)
        if not result:
            return 0, None

        score = result["score_adjustment"]
        probability = result["spawn_probability"]
        return score, f"duel spawn model {probability:.2f}"

    def confidence(self, best_score, runner_up_score):
        gap = max(0, best_score - runner_up_score)
        return round(min(0.95, 0.35 + gap / 80), 2)
