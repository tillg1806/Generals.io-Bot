from config import LOOP_DETECTION_MEMORY
from prioritized_move import PrioritizedMove


class StrategyPriorityAgent:
    def prioritize(self, move, base_score, recent_moves, flow_chain):
        if base_score is None:
            return None

        tactical = self.is_tactical(move)
        score = base_score
        reasons = []

        reverse_penalty = self.reverse_penalty(move, recent_moves, tactical)
        if reverse_penalty is None:
            return None
        if reverse_penalty:
            score -= reverse_penalty
            reasons.append(f"reverse -{reverse_penalty}")

        edge_penalty = self.edge_reuse_penalty(move, recent_moves, tactical)
        if edge_penalty is None:
            return None
        if edge_penalty:
            score -= edge_penalty
            reasons.append(f"edge reuse -{edge_penalty}")

        stale_stack_penalty = self.stale_stack_penalty(move, recent_moves, tactical)
        if stale_stack_penalty is None:
            return None
        if stale_stack_penalty:
            score -= stale_stack_penalty
            reasons.append(f"fresh target pullback -{stale_stack_penalty}")

        idle_owned_penalty = self.idle_owned_tile_penalty(move, tactical)
        if idle_owned_penalty is None:
            return None
        if idle_owned_penalty:
            score -= idle_owned_penalty
            reasons.append(f"idle own tile -{idle_owned_penalty}")

        flow_bonus = self.flow_bonus(move, flow_chain)
        if flow_bonus:
            score += flow_bonus
            reasons.append(f"flow +{flow_bonus}")

        return PrioritizedMove(score=score, reason=", ".join(reasons))

    def is_tactical(self, move):
        return (
            move["defends_general"]
            or move["attacks_threat"]
            or move["target_is_enemy_general"]
            or move["target_is_enemy_tile"]
            or move["can_take_city"]
            or move.get("reinforcement_attack_push", False)
        )

    def has_forward_progress(self, move):
        return (
            move["target_is_new_tile"]
            or move["opens_route_gateway"]
            or move["follows_city_free_route"]
            or move["follows_fallback_route"]
            or move["route_progress"] > 0
            or move["fallback_route_progress"] > 0
            or move["moves_toward_city"]
            or move["dominance_route_push"]
            or move["extends_flow_chain"]
            or move["drains_overstacked_along_flow"]
            or move.get("stack_consolidation", False)
            or move.get("reinforcement_route_push", False)
            or move.get("reinforcement_frontier_push", False)
        )

    def reverse_penalty(self, move, recent_moves, tactical):
        source = move["source"]
        target = move["target"]
        if not recent_moves or recent_moves[-1] != (target, source):
            return 0

        if tactical:
            return 35000
        if self.has_forward_progress(move):
            return 55000
        return None

    def edge_reuse_penalty(self, move, recent_moves, tactical):
        source = move["source"]
        target = move["target"]
        edge = frozenset((source, target))
        repeats = sum(
            1
            for recent_source, recent_target in recent_moves[-LOOP_DETECTION_MEMORY:]
            if frozenset((recent_source, recent_target)) == edge
        )
        if repeats == 0:
            return 0

        if repeats >= 2 and not tactical and not move["target_is_new_tile"]:
            return None

        penalty = repeats * 18000
        if tactical:
            return penalty // 2
        if move["target_is_new_tile"]:
            return penalty // 3
        return penalty

    def stale_stack_penalty(self, move, recent_moves, tactical):
        source = move["source"]
        if tactical:
            return 0

        recent_targets = [target for _, target in recent_moves[-4:]]
        if source not in recent_targets:
            return 0

        if move["target_is_new_tile"] or move["opens_route_gateway"]:
            return 0
        if move["route_progress"] > 0 or move["fallback_route_progress"] > 0:
            return 12000
        return None

    def idle_owned_tile_penalty(self, move, tactical):
        if tactical:
            return 0
        if move["target_is_new_tile"] or move["target_is_enemy_tile"]:
            return 0
        if self.has_forward_progress(move):
            return 0
        return None

    def flow_bonus(self, move, flow_chain):
        if not flow_chain:
            return 0

        source = move["source"]
        target = move["target"]
        last_source, last_target = flow_chain[-1]

        if source == last_target and target != last_source:
            return 6000

        if any(
            source == recent_target and target != recent_source
            for recent_source, recent_target in flow_chain[-4:]
        ):
            return 3000

        return 0
