from config import (
    AGGRESSION_MIN_CHAIN_ARMY,
    AGGRESSION_PULSE_INTERVAL,
    CENTER_REACHED_DISTANCE,
    CITY_ATTACK_BUFFER,
    DOMINANCE_CITY_PUSH_BONUS,
    DOMINANCE_ARMY_RATIO,
    DOMINANCE_ROUTE_PUSH_BONUS,
    ENEMY_PREDICTION_MIN_HEAT,
    EXPAND_AFTER_TURN,
    FLOW_CHAIN_MEMORY,
    FLOW_CHAIN_MIN_ARMY,
    GENERAL_DEFENSE_RADIUS,
    GENERAL_MIN_ARMY_RESERVE,
    GENERAL_SEARCH_RADIUS,
    GENERAL_SEARCH_FRONTIER_BONUS_DISTANCE,
    HIGH_ENEMY_ARMY,
    INFORMATION_BELIEF_PROXIMITY_BONUS,
    INFORMATION_GAIN_BONUS,
    KNOWN_GENERAL_ROUTE_BONUS,
    KNOWN_GENERAL_TARGET_BONUS,
    LOOKAHEAD_BEAM_WIDTH,
    LOOKAHEAD_SCORE_CAP,
    LOOKAHEAD_SCORE_WEIGHT,
    ACTION_VALUE_BEAM_WIDTH,
    ACTION_VALUE_SCORE_CAP,
    MOVE_EXPLANATION_TOP_K,
    LOOP_DETECTION_MEMORY,
    LOOP_TARGET_VISIT_LIMIT,
    MIN_CITY_ATTACK_ARMY,
    NORMAL_TILE_ARMY_SOFT_CAP,
    NORMAL_TILE_ARMY_SOFT_CAP_MAX,
    NORMAL_TILE_ARMY_SOFT_CAP_START_TURN,
    NORMAL_TILE_ARMY_SOFT_CAP_STEP_TURNS,
    OWN_CITY_ARMY_RESERVE,
    OWN_CITY_PUSH_ARMY,
    RESCOUT_AFTER_VISIBLE_TURNS,
    REINFORCEMENT_ATTACK_BONUS,
    REINFORCEMENT_ATTACK_LEAD_TURNS,
    REINFORCEMENT_EXPAND_WINDOW,
    REINFORCEMENT_FRONTIER_BONUS,
    REINFORCEMENT_INTERVAL,
    REINFORCEMENT_ROUTE_BONUS,
    REINFORCEMENT_STAGING_WINDOW,
    REINFORCEMENT_WAYPOINT_LOOKAHEAD,
    ROUTE_DISTANCE_PENALTY,
    ROUTE_GATEWAY_BONUS,
    ROUTE_PROGRESS_BONUS,
    SPAWN_TARGET_AFTER_TURN,
    STACK_CONSOLIDATION_BONUS,
    STACK_CONSOLIDATION_MIN_ARMY,
    STACK_CONSOLIDATION_RADIUS,
    STALEMATE_BREAKOUT_VISIBLE_TURNS,
    STALEMATE_CITY_BONUS,
    STALEMATE_MIN_TARGET_REPEATS,
    STALEMATE_MIN_VISIBLE_TURN,
    STALEMATE_NEW_TILE_BONUS,
    STALEMATE_REPEAT_PENALTY,
    STALEMATE_ROUTE_BONUS,
    STALEMATE_SCOUT_BONUS,
    STALEMATE_WINDOW_VISIBLE_TURNS,
    TARGET_COMMITMENT_MAX_VISIBLE_TURNS,
    TARGET_COMMITMENT_REACHED_DISTANCE,
    TILE_CATCHUP_FACTOR,
    FALLBACK_ROUTE_PROGRESS_BONUS,
)
from pathfinding import (
    build_distance_map,
    distance_to_center,
    distance_to_target,
    is_passable,
    neighbors,
    reachable_tiles_from,
    xy_to_index,
)
from priority_agent import StrategyPriorityAgent
from move import Move
from move_utility import MoveUtility
from reinforcement_context import ReinforcementContext
from stalemate_state import StalemateState
from strategy_option import StrategyOption


class Strategy:
    def __init__(
        self,
        state,
        reserve_after_turn,
        city_focus_after_turn,
        general_attack_after_turn,
        logger=None,
    ):
        self.state = state
        self.reserve_after_turn = reserve_after_turn
        self.city_focus_after_turn = city_focus_after_turn
        self.general_attack_after_turn = general_attack_after_turn
        self.logger = logger
        self.planned_target = None
        self.planned_distances = {}
        self.planned_fallback_distances = {}
        self.committed_target = None
        self.committed_since_visible_turn = None
        self.committed_reason = None
        self.initial_enemy_general_guess = None
        self.recent_moves = []
        self.flow_chain = []
        self.priority_agent = StrategyPriorityAgent()
        self.enemy_general_beliefs = []
        self.stalemate = StalemateState()
        self.current_option = StrategyOption()
        self.last_selected_move = None
        self.last_move_explanations = []
        self.action_value_agent = None
        self.coach_bias = {
            "expansion_bias": 0.0,
            "city_bias": 0.0,
            "attack_bias": 0.0,
            "defense_bias": 0.0,
            "route_bias": 0.0,
        }

    def log(self, *parts):
        if self.logger:
            self.logger(*parts)

    def can_attack_city(self, source, target, armies):
        if target not in self.state.city_set():
            return True

        needed_army = max(armies[target] + CITY_ATTACK_BUFFER, MIN_CITY_ATTACK_ARMY)
        return armies[source] > needed_army

    def choose_reachable_proxy_target(self, desired_target, terrain):
        if desired_target is None:
            return None

        reachable = reachable_tiles_from(
            self.state,
            self.state.my_general_index,
            terrain,
            avoid_cities=True,
        )
        if not reachable:
            return desired_target

        if desired_target in reachable and is_passable(terrain[desired_target]):
            return desired_target

        return min(
            reachable,
            key=lambda index: (
                distance_to_target(self.state, index, desired_target),
                distance_to_center(self.state, index),
            ),
        )

    def choose_strategy_target(self, terrain, turn, army_dominance=False):
        if self.state.enemy_general_index is not None:
            return self.choose_reachable_proxy_target(self.state.enemy_general_index, terrain)

        if army_dominance:
            dominance_target = self.choose_dominance_target(terrain)
            if dominance_target is not None:
                return self.choose_reachable_proxy_target(dominance_target, terrain)

        if self.state.visible_enemy_tiles:
            enemy_target = min(
                self.state.visible_enemy_tiles,
                key=lambda index: distance_to_center(self.state, index),
            )
            return self.choose_reachable_proxy_target(enemy_target, terrain)

        belief_target = self.choose_enemy_general_belief_target(terrain)
        if belief_target is not None:
            return self.choose_reachable_proxy_target(belief_target, terrain)

        if self.initial_enemy_general_guess is not None:
            return self.choose_reachable_proxy_target(self.initial_enemy_general_guess, terrain)

        search_target = self.choose_enemy_general_search_target(terrain, turn)
        if search_target is not None:
            return self.choose_reachable_proxy_target(search_target, terrain)

        if (
            self.has_reached_center(terrain)
            and self.state.visible_turn(turn) >= SPAWN_TARGET_AFTER_TURN
        ):
            spawn_target = self.choose_spawn_target(terrain)
            if spawn_target is not None:
                return self.choose_reachable_proxy_target(spawn_target, terrain)

        center_x = self.state.width // 2
        center_y = self.state.height // 2
        return self.choose_reachable_proxy_target(
            xy_to_index(self.state, center_x, center_y),
            terrain,
        )

    def choose_committed_strategy_target(self, terrain, turn, army_dominance=False):
        visible_turn = self.state.visible_turn(turn)
        desired_target = self.choose_strategy_target(terrain, turn, army_dominance)

        if self.should_keep_committed_target(terrain, visible_turn):
            return self.committed_target

        self.committed_target = desired_target
        self.committed_since_visible_turn = visible_turn
        self.committed_reason = "strategy"
        return self.committed_target

    def should_keep_committed_target(self, terrain, visible_turn):
        if self.committed_target is None:
            return False
        if not self.is_committed_target_valid(terrain):
            return False
        if self.has_reached_committed_target(terrain):
            return False
        if self.committed_since_visible_turn is None:
            return True

        return (
            visible_turn - self.committed_since_visible_turn
            <= TARGET_COMMITMENT_MAX_VISIBLE_TURNS
        )

    def is_committed_target_valid(self, terrain):
        if self.committed_target >= len(terrain):
            return False
        if not is_passable(terrain[self.committed_target]):
            return False

        reachable = reachable_tiles_from(
            self.state,
            self.state.my_general_index,
            terrain,
            avoid_cities=True,
        )
        if not reachable:
            return True

        return self.committed_target in reachable

    def has_reached_committed_target(self, terrain):
        if self.committed_target is None or self.committed_target >= len(terrain):
            return True
        if terrain[self.committed_target] == self.state.player_index:
            return True

        closest_owned = self.owned_tile_closest_to_target(terrain, self.committed_target)
        if closest_owned is None:
            return False

        return (
            distance_to_target(self.state, closest_owned, self.committed_target)
            <= TARGET_COMMITMENT_REACHED_DISTANCE
        )

    def choose_dominance_target(self, terrain):
        suspected_target = self.choose_enemy_movement_target(terrain)
        if suspected_target is not None:
            return self.project_target_to_far_edge(suspected_target, terrain)

        if self.state.visible_enemy_tiles:
            enemy_target = max(
                self.state.visible_enemy_tiles,
                key=lambda index: distance_to_target(self.state, self.state.my_general_index, index),
            )
            return self.project_target_to_far_edge(enemy_target, terrain)

        spawn_target = self.choose_spawn_target(terrain)
        if spawn_target is not None:
            return self.project_target_to_far_edge(spawn_target, terrain)

        return None

    def choose_enemy_movement_target(self, terrain):
        if self.state.my_general_index is None:
            return None

        candidates = [
            index
            for index, heat in self.state.enemy_movement_heat.items()
            if heat > 0 and index < len(terrain) and is_passable(terrain[index])
        ]
        if not candidates:
            return None

        return max(
            candidates,
            key=lambda index: (
                self.state.enemy_movement_heat.get(index, 0),
                distance_to_target(self.state, self.state.my_general_index, index),
            ),
        )

    def project_target_to_far_edge(self, target, terrain):
        if (
            self.state.my_general_index is None
            or target is None
            or self.state.width <= 0
            or self.state.height <= 0
        ):
            return target

        origin_x = self.state.my_general_index % self.state.width
        origin_y = self.state.my_general_index // self.state.width
        target_x = target % self.state.width
        target_y = target // self.state.width
        direction_x = target_x - origin_x
        direction_y = target_y - origin_y

        if direction_x == 0 and direction_y == 0:
            return target

        edge_tiles = []
        for index in range(len(terrain)):
            x = index % self.state.width
            y = index // self.state.width
            if x not in (0, self.state.width - 1) and y not in (0, self.state.height - 1):
                continue
            if is_passable(terrain[index]):
                edge_tiles.append(index)

        if not edge_tiles:
            return target

        return max(
            edge_tiles,
            key=lambda index: (
                ((index % self.state.width) - origin_x) * direction_x
                + ((index // self.state.width) - origin_y) * direction_y,
                -distance_to_target(self.state, index, target),
            ),
        )

    def set_enemy_general_beliefs(self, beliefs):
        self.enemy_general_beliefs = [
            belief
            for belief in beliefs or []
            if isinstance(belief, dict) and belief.get("index") is not None
        ]
        if self.enemy_general_beliefs:
            self.initial_enemy_general_guess = self.enemy_general_beliefs[0]["index"]

    def valid_enemy_general_beliefs(self, terrain):
        valid = []
        for belief in self.enemy_general_beliefs:
            index = belief.get("index")
            if index is None or index >= len(terrain):
                continue
            if terrain[index] == self.state.player_index:
                continue
            if not is_passable(terrain[index]):
                continue
            if self.state.has_seen(index):
                continue
            if index in self.state.city_set():
                continue
            valid.append(belief)
        return valid

    def choose_enemy_general_belief_target(self, terrain):
        beliefs = self.valid_enemy_general_beliefs(terrain)
        if not beliefs:
            return None

        return max(
            beliefs,
            key=lambda belief: (
                float(belief.get("belief") or 0.0),
                float(belief.get("score") or 0.0),
            ),
        )["index"]

    def choose_enemy_general_search_target(self, terrain, turn):
        visible_turn = self.state.visible_turn(turn)
        search_centers = [
            belief["index"]
            for belief in self.valid_enemy_general_beliefs(terrain)[:5]
        ]
        if not search_centers:
            search_centers = self.possible_spawn_targets()

        targets = [
            target
            for spawn_target in search_centers
            for target in self.search_area_targets(spawn_target, terrain)
        ]
        if not targets:
            return None

        closest_owned = self.owned_tile_closest_to_center(terrain)
        if closest_owned is None:
            closest_owned = self.state.my_general_index

        return min(
            targets,
            key=lambda target: (
                self.enemy_general_search_target_score(target, terrain, visible_turn),
                distance_to_target(self.state, closest_owned, target),
            ),
        )

    def search_area_targets(self, center, terrain):
        if center >= len(terrain):
            return []

        targets = []
        for index in range(len(terrain)):
            if terrain[index] == self.state.player_index:
                continue
            if not is_passable(terrain[index]):
                continue
            if distance_to_target(self.state, index, center) <= GENERAL_SEARCH_RADIUS:
                targets.append(index)

        return targets

    def enemy_general_search_target_score(self, target, terrain, visible_turn):
        if not self.state.has_seen(target):
            return -100

        last_seen = self.state.visible_turn(self.state.last_seen(target))
        if last_seen >= 0 and visible_turn - last_seen >= RESCOUT_AFTER_VISIBLE_TURNS:
            return -50

        if terrain[target] >= 0:
            return 100

        return 0

    def should_catch_up_tiles(self):
        enemy_tiles = self.state.biggest_enemy_tile_count()
        if enemy_tiles == 0:
            return False

        return self.state.my_tile_count() < enemy_tiles * TILE_CATCHUP_FACTOR

    def has_army_dominance(self):
        enemy_total = self.state.biggest_enemy_total_army()
        if enemy_total <= 0:
            return False

        return self.state.my_total_army() >= enemy_total * DOMINANCE_ARMY_RATIO

    def enemy_threat_tiles(self, terrain, armies):
        if self.state.my_general_index is None or self.state.player_index is None:
            return []

        threats = []
        for index, owner in enumerate(terrain):
            if owner < 0 or owner == self.state.player_index:
                continue
            if armies[index] < HIGH_ENEMY_ARMY:
                continue
            if distance_to_target(self.state, index, self.state.my_general_index) <= GENERAL_DEFENSE_RADIUS:
                threats.append(index)

        for index, heat in self.state.enemy_prediction_heat.items():
            if heat < ENEMY_PREDICTION_MIN_HEAT:
                continue
            if index >= len(terrain) or not is_passable(terrain[index]):
                continue
            if terrain[index] == self.state.player_index and armies[index] > GENERAL_MIN_ARMY_RESERVE:
                continue
            if distance_to_target(self.state, index, self.state.my_general_index) <= GENERAL_DEFENSE_RADIUS:
                threats.append(index)

        return threats

    def choose_defense_target(self, terrain, armies):
        threats = self.enemy_threat_tiles(terrain, armies)
        if not threats:
            return None

        return min(
            threats,
            key=lambda index: (
                distance_to_target(self.state, index, self.state.my_general_index),
                -self.state.enemy_prediction_heat.get(index, 0),
                -armies[index],
            ),
        )

    def owned_tile_closest_to_center(self, terrain):
        owned_tiles = [
            index
            for index, owner in enumerate(terrain)
            if owner == self.state.player_index
        ]
        if not owned_tiles:
            return None

        return min(owned_tiles, key=lambda index: distance_to_center(self.state, index))

    def owned_tile_closest_to_target(self, terrain, target):
        owned_tiles = [
            index
            for index, owner in enumerate(terrain)
            if owner == self.state.player_index
        ]
        if not owned_tiles:
            return None

        return min(
            owned_tiles,
            key=lambda index: distance_to_target(self.state, index, target),
        )

    def has_reached_center(self, terrain):
        closest_owned = self.owned_tile_closest_to_center(terrain)
        if closest_owned is None:
            return False

        return distance_to_center(self.state, closest_owned) <= CENTER_REACHED_DISTANCE

    def possible_spawn_targets(self):
        if self.state.my_general_index is None or self.state.width == 0 or self.state.height == 0:
            return []

        x, y = self.state.my_general_index % self.state.width, self.state.my_general_index // self.state.width
        candidates = [
            xy_to_index(self.state, self.state.width - 1 - x, self.state.height - 1 - y),
            xy_to_index(self.state, self.state.width - 1 - x, y),
            xy_to_index(self.state, x, self.state.height - 1 - y),
            xy_to_index(self.state, 0, 0),
            xy_to_index(self.state, self.state.width - 1, 0),
            xy_to_index(self.state, 0, self.state.height - 1),
            xy_to_index(self.state, self.state.width - 1, self.state.height - 1),
        ]

        unique_candidates = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)

        return unique_candidates

    def choose_spawn_target(self, terrain):
        targets = [
            target
            for target in self.possible_spawn_targets()
            if terrain[target] != self.state.player_index and is_passable(terrain[target])
        ]
        if not targets:
            targets = self.possible_spawn_targets()

        if not targets:
            return None

        closest_owned = self.owned_tile_closest_to_center(terrain)
        if closest_owned is None:
            closest_owned = self.state.my_general_index

        return min(
            targets,
            key=lambda target: distance_to_target(self.state, closest_owned, target),
        )

    def frontier_score(self, index, terrain):
        score = 0

        for neighbor in neighbors(self.state, index):
            if not is_passable(terrain[neighbor]):
                continue
            if terrain[neighbor] != self.state.player_index:
                score += 1

        return score

    def search_frontier_score(self, index, terrain, strategy_target):
        if self.state.enemy_general_index is not None:
            return 0
        if strategy_target is None:
            return 0
        if terrain[index] == self.state.player_index:
            return 0

        distance = distance_to_target(self.state, index, strategy_target)
        if distance > GENERAL_SEARCH_FRONTIER_BONUS_DISTANCE:
            return 0

        score = GENERAL_SEARCH_FRONTIER_BONUS_DISTANCE - distance + 1
        if not self.state.has_seen(index):
            score += 8

        return score

    def information_gain_score(self, index, terrain, visible_turn):
        if index >= len(terrain):
            return 0

        score = 0
        if not self.state.has_seen(index):
            score += 6

        for neighbor in neighbors(self.state, index):
            if neighbor >= len(terrain) or not is_passable(terrain[neighbor]):
                continue
            if not self.state.has_seen(neighbor):
                score += 2
                continue
            last_seen = self.state.visible_turn(self.state.last_seen(neighbor))
            if last_seen >= 0 and visible_turn - last_seen >= RESCOUT_AFTER_VISIBLE_TURNS:
                score += 1

        return score

    def belief_proximity_score(self, index):
        beliefs = self.enemy_general_beliefs[:5]
        if not beliefs:
            return 0

        best = 0.0
        for belief in beliefs:
            belief_index = belief.get("index")
            if belief_index is None:
                continue
            distance = distance_to_target(self.state, index, belief_index)
            if distance > GENERAL_SEARCH_RADIUS + 3:
                continue
            value = float(belief.get("belief") or 0.0) * max(0, GENERAL_SEARCH_RADIUS + 4 - distance)
            best = max(best, value)
        return int(best * 10)

    def nearest_unowned_city_distance(self, index, terrain):
        unowned_cities = [
            city
            for city in self.state.city_set()
            if city < len(terrain) and terrain[city] != self.state.player_index
        ]
        if not unowned_cities:
            return None

        return min(
            distance_to_target(self.state, index, city)
            for city in unowned_cities
        )

    def normal_tile_soft_cap(self, turn):
        visible_turn = self.state.visible_turn(turn)
        if visible_turn < NORMAL_TILE_ARMY_SOFT_CAP_START_TURN:
            return NORMAL_TILE_ARMY_SOFT_CAP

        steps = (
            (visible_turn - NORMAL_TILE_ARMY_SOFT_CAP_START_TURN)
            // NORMAL_TILE_ARMY_SOFT_CAP_STEP_TURNS
        )
        return min(
            NORMAL_TILE_ARMY_SOFT_CAP + steps,
            NORMAL_TILE_ARMY_SOFT_CAP_MAX,
        )

    def should_reserve_general_army(self, source, source_army, turn):
        if source != self.state.my_general_index:
            return False
        if self.state.visible_turn(turn) < self.reserve_after_turn:
            return False

        return source_army <= GENERAL_MIN_ARMY_RESERVE

    def should_half_move_from_general(self, source, source_army, turn):
        if source != self.state.my_general_index:
            return False
        if self.state.visible_turn(turn) < self.reserve_after_turn:
            return False

        return source_army > GENERAL_MIN_ARMY_RESERVE

    def reinforcement_context(self, turn, strategy_target, terrain, armies):
        if turn < 0 or REINFORCEMENT_INTERVAL <= 0:
            return ReinforcementContext(0, 0, False, False, False)

        phase = turn % REINFORCEMENT_INTERVAL
        turns_until = (REINFORCEMENT_INTERVAL - phase) % REINFORCEMENT_INTERVAL
        turns_since = phase
        staging = 0 < turns_until <= REINFORCEMENT_STAGING_WINDOW
        attack_window = 0 < turns_until <= REINFORCEMENT_ATTACK_LEAD_TURNS
        expand_window = turns_since <= REINFORCEMENT_EXPAND_WINDOW
        waypoint = None

        if staging and strategy_target is not None:
            waypoint = self.choose_reinforcement_waypoint(
                strategy_target,
                terrain,
                armies,
                turns_until,
            )

        return ReinforcementContext(
            turns_until=turns_until,
            turns_since=turns_since,
            staging=staging,
            attack_window=attack_window,
            expand_window=expand_window,
            waypoint=waypoint,
        )

    def choose_reinforcement_waypoint(self, strategy_target, terrain, armies, turns_until):
        if self.state.my_general_index is None or strategy_target is None:
            return None
        if strategy_target >= len(terrain) or not is_passable(terrain[strategy_target]):
            return None

        owned_tiles = [
            index
            for index, owner in enumerate(terrain)
            if owner == self.state.player_index and armies[index] > 1
        ]
        if not owned_tiles:
            return None

        staging_source = min(
            owned_tiles,
            key=lambda index: (
                distance_to_target(self.state, index, strategy_target),
                -armies[index],
            ),
        )
        if staging_source == strategy_target:
            return None

        route_distances = build_distance_map(
            self.state,
            strategy_target,
            terrain,
            avoid_cities=True,
        )
        if staging_source not in route_distances:
            route_distances = build_distance_map(
                self.state,
                strategy_target,
                terrain,
                avoid_cities=False,
            )
        if staging_source not in route_distances:
            return None

        steps = max(
            1,
            min(
                REINFORCEMENT_WAYPOINT_LOOKAHEAD,
                max(1, turns_until - REINFORCEMENT_ATTACK_LEAD_TURNS),
                route_distances[staging_source],
            ),
        )
        current = staging_source
        for _ in range(steps):
            next_steps = [
                neighbor
                for neighbor in neighbors(self.state, current)
                if neighbor in route_distances
                and route_distances[neighbor] < route_distances[current]
                and is_passable(terrain[neighbor])
            ]
            if not next_steps:
                break
            current = min(
                next_steps,
                key=lambda index: (
                    route_distances[index],
                    terrain[index] == self.state.player_index,
                ),
            )

        return current if current != staging_source else None

    def update_planned_routes(self, strategy_target, terrain):
        self.planned_target = strategy_target
        self.planned_distances = build_distance_map(
            self.state,
            strategy_target,
            terrain,
            avoid_cities=True,
        )
        self.planned_fallback_distances = build_distance_map(
            self.state,
            strategy_target,
            terrain,
            avoid_cities=False,
        )

        self.log(
            "Route planned:",
            "Target =", strategy_target,
            "Mode =", self.committed_reason,
            "city-free reachable tiles =", len(self.planned_distances),
            "Fallback-Tiles =", len(self.planned_fallback_distances),
        )

    def update_stalemate_state(self, visible_turn, terrain, strategy_target):
        if visible_turn < STALEMATE_MIN_VISIBLE_TURN:
            return

        my_tiles = self.state.my_tile_count()
        seen_tiles = len(self.state.seen_tiles)
        target_distance = self.closest_owned_distance_to_target(terrain, strategy_target)
        stalemate = self.stalemate

        if stalemate.last_progress_visible_turn is None:
            stalemate.last_progress_visible_turn = visible_turn
            stalemate.last_tile_count = my_tiles
            stalemate.last_seen_count = seen_tiles
            stalemate.last_target_distance = target_distance
            stalemate.last_strategy_target = strategy_target
            return

        if strategy_target == stalemate.last_strategy_target:
            stalemate.repeated_target_count += 1
        else:
            stalemate.repeated_target_count = 0
            stalemate.last_strategy_target = strategy_target

        made_progress = my_tiles > stalemate.last_tile_count or seen_tiles > stalemate.last_seen_count
        if (
            target_distance is not None
            and stalemate.last_target_distance is not None
            and target_distance < stalemate.last_target_distance - 1
        ):
            made_progress = True

        if made_progress:
            stalemate.last_progress_visible_turn = visible_turn
            stalemate.last_tile_count = my_tiles
            stalemate.last_seen_count = seen_tiles
            stalemate.last_target_distance = target_distance
            if stalemate.active:
                self.log(
                    "Stalemate breakout resolved:",
                    "turn =", visible_turn,
                    "tiles =", my_tiles,
                    "seen =", seen_tiles,
                )
            stalemate.active = False
            stalemate.activated_at_visible_turn = None
            stalemate.reason = None
            return

        if target_distance is not None:
            stalemate.last_target_distance = target_distance

        if self.state.enemy_general_index is not None:
            return

        stalled_for = visible_turn - stalemate.last_progress_visible_turn
        repeated_target = stalemate.repeated_target_count >= STALEMATE_MIN_TARGET_REPEATS
        should_activate = (
            not stalemate.active
            and stalled_for >= STALEMATE_WINDOW_VISIBLE_TURNS
            and repeated_target
        )
        if should_activate:
            stalemate.active = True
            stalemate.activated_at_visible_turn = visible_turn
            stalemate.reason = (
                f"no progress for {stalled_for} visible turns, "
                f"target repeated {stalemate.repeated_target_count} times"
            )
            self.log("Stalemate breakout active:", stalemate.reason)
            return

        if (
            stalemate.active
            and stalemate.activated_at_visible_turn is not None
            and visible_turn - stalemate.activated_at_visible_turn > STALEMATE_BREAKOUT_VISIBLE_TURNS
        ):
            stalemate.active = False
            stalemate.activated_at_visible_turn = None
            stalemate.last_progress_visible_turn = visible_turn
            stalemate.reason = None
            self.log("Stalemate breakout cooled down:", "turn =", visible_turn)

    def closest_owned_distance_to_target(self, terrain, target):
        if target is None or target >= len(terrain):
            return None

        closest_owned = self.owned_tile_closest_to_target(terrain, target)
        if closest_owned is None:
            return None

        return distance_to_target(self.state, closest_owned, target)

    def update_strategy_option(
        self,
        visible_turn,
        defense_target,
        strategy_target,
        terrain,
        army_dominance,
        catch_up_tiles,
    ):
        if defense_target is not None:
            name = "secure_general"
            reason = "defense target visible"
        elif self.state.enemy_general_index is not None:
            name = "attack_general"
            reason = "enemy general known"
        elif self.stalemate.active:
            name = "break_stalemate"
            reason = self.stalemate.reason
        elif self.valid_enemy_general_beliefs(terrain):
            name = "scout_general"
            reason = "enemy general belief candidates"
        elif self.has_unowned_city_targets(terrain) and visible_turn >= self.city_focus_after_turn:
            name = "contest_city"
            reason = "city focus enabled"
        elif army_dominance:
            name = "press_route"
            reason = "army dominance"
        elif catch_up_tiles:
            name = "expand"
            reason = "tile catch-up"
        elif strategy_target is not None:
            name = "press_route"
            reason = "committed route"
        else:
            name = "expand"
            reason = "default"

        if name != self.current_option.name:
            self.current_option = StrategyOption(
                name=name,
                reason=reason,
                started_at_visible_turn=visible_turn,
            )
            self.log("Option:", name, "reason =", reason)
        else:
            self.current_option.reason = reason

    def has_unowned_city_targets(self, terrain):
        return any(
            city < len(terrain) and terrain[city] != self.state.player_index
            for city in self.state.city_set()
        )

    def choose_move(self, turn):
        if self.state.player_index is None or self.state.width == 0 or self.state.height == 0:
            return None

        terrain, armies = self.state.split_map()
        if len(terrain) < self.state.width * self.state.height or len(armies) < self.state.width * self.state.height:
            return None

        defense_target = self.choose_defense_target(terrain, armies)
        army_dominance = self.has_army_dominance()
        strategy_target = defense_target
        if strategy_target is not None:
            self.committed_reason = "defense"
        elif self.state.enemy_general_index is not None:
            strategy_target = self.choose_reachable_proxy_target(self.state.enemy_general_index, terrain)
            self.committed_target = strategy_target
            self.committed_since_visible_turn = self.state.visible_turn(turn)
            self.committed_reason = "enemy_general"
        else:
            strategy_target = self.choose_committed_strategy_target(terrain, turn, army_dominance)

        reinforcement = self.reinforcement_context(turn, strategy_target, terrain, armies)
        route_target = strategy_target
        if (
            defense_target is None
            and self.state.enemy_general_index is None
            and reinforcement.waypoint is not None
        ):
            route_target = reinforcement.waypoint
            self.committed_reason = "reinforcement_waypoint"
            self.log(
                "Reinforcement waypoint:",
                route_target,
                "final target =",
                strategy_target,
                "turns until reinforcement =",
                reinforcement.turns_until,
            )

        self.update_planned_routes(route_target, terrain)

        if (
            not self.state.expansion_started
            and self.state.visible_turn(turn) < EXPAND_AFTER_TURN
        ):
            my_general_army = armies[self.state.my_general_index]
            self.log(
                f"Waiting until turn {EXPAND_AFTER_TURN}: "
                f"turn {self.state.visible_turn(turn)}, general has {my_general_army} units"
            )
            return None

        self.state.expansion_started = True
        return self._choose_ranked_move(
            turn,
            terrain,
            armies,
            defense_target,
            route_target,
            reinforcement,
        )

    def _choose_ranked_move(self, turn, terrain, armies, defense_target, strategy_target, reinforcement):
        army_dominance = self.has_army_dominance()
        catch_up_tiles = self.should_catch_up_tiles()
        visible_turn = self.state.visible_turn(turn)
        enemy_general_known = self.state.enemy_general_index is not None
        self.update_strategy_option(
            visible_turn,
            defense_target,
            strategy_target,
            terrain,
            army_dominance,
            catch_up_tiles,
        )
        city_focus_enabled = visible_turn >= self.city_focus_after_turn
        general_attack_enabled = (
            enemy_general_known
            or army_dominance
            or visible_turn >= self.general_attack_after_turn
        )
        normal_tile_soft_cap = self.normal_tile_soft_cap(turn)
        self.update_stalemate_state(visible_turn, terrain, strategy_target)
        stalemate_breakout = self.stalemate.active and defense_target is None
        aggression_pulse = (
            defense_target is None
            and visible_turn >= EXPAND_AFTER_TURN
            and (
                visible_turn % AGGRESSION_PULSE_INTERVAL == 0
                or reinforcement.expand_window
                or stalemate_breakout
            )
        )
        broad_expansion_mode = (
            defense_target is None
            and (
                catch_up_tiles
                or (self.has_reached_center(terrain) and not self.state.visible_enemy_tiles)
                or stalemate_breakout
            )
        )

        moves = []
        skipped_cities = 0
        blocked_tiles = 0
        own_movable_tiles = 0

        for source, owner in enumerate(terrain):
            if owner != self.state.player_index or armies[source] <= 1:
                continue
            if (
                self.should_reserve_general_army(source, armies[source], turn)
                and not army_dominance
                and not enemy_general_known
            ):
                continue

            own_movable_tiles += 1
            source_distance = distance_to_target(self.state, source, strategy_target)
            source_city_distance = self.nearest_unowned_city_distance(source, terrain)
            planned_source_distance = self.planned_distances.get(source)
            fallback_source_distance = self.planned_fallback_distances.get(source)

            for target in neighbors(self.state, source):
                if not is_passable(terrain[target]):
                    blocked_tiles += 1
                    continue
                target_is_city = target in self.state.city_set()
                can_take_city = target_is_city and self.can_attack_city(source, target, armies)
                if target_is_city and not can_take_city:
                    skipped_cities += 1
                    continue

                target_is_new_tile = terrain[target] != self.state.player_index
                target_is_enemy_tile = terrain[target] >= 0 and terrain[target] != self.state.player_index
                target_is_enemy_general = target == self.state.enemy_general_index
                target_is_priority_enemy_general = general_attack_enabled and target_is_enemy_general
                if target_is_enemy_tile and armies[source] <= armies[target] + 1:
                    continue

                planned_target_distance = self.planned_distances.get(target)
                fallback_target_distance = self.planned_fallback_distances.get(target)
                target_distance = planned_target_distance
                if target_distance is None:
                    target_distance = fallback_target_distance
                if target_distance is None:
                    target_distance = distance_to_target(self.state, target, strategy_target)
                search_frontier_score = self.search_frontier_score(target, terrain, strategy_target)
                information_gain = self.information_gain_score(target, terrain, visible_turn)
                belief_proximity = self.belief_proximity_score(target)

                follows_city_free_route = (
                    planned_source_distance is not None
                    and planned_target_distance is not None
                    and planned_target_distance < planned_source_distance
                    and (defense_target is not None or not can_take_city)
                )
                follows_fallback_route = (
                    fallback_source_distance is not None
                    and fallback_target_distance is not None
                    and fallback_target_distance < fallback_source_distance
                )
                route_progress = 0
                if planned_source_distance is not None and planned_target_distance is not None:
                    route_progress = planned_source_distance - planned_target_distance

                fallback_route_progress = 0
                if fallback_source_distance is not None and fallback_target_distance is not None:
                    fallback_route_progress = fallback_source_distance - fallback_target_distance

                opens_route_gateway = (
                    target_is_new_tile
                    and (
                        (
                            planned_source_distance is None
                            and planned_target_distance is not None
                        )
                        or (
                            fallback_source_distance is None
                            and fallback_target_distance is not None
                        )
                    )
                )
                is_closer = distance_to_target(self.state, target, strategy_target) < source_distance
                target_city_distance = self.nearest_unowned_city_distance(target, terrain)
                defends_general = defense_target is not None and target_distance < source_distance
                attacks_threat = target == defense_target
                attacks_known_general_route = (
                    enemy_general_known
                    and defense_target is None
                    and (
                        target_is_enemy_general
                        or follows_city_free_route
                        or follows_fallback_route
                        or is_closer
                    )
                )
                source_is_owned_city = source in self.state.city_set()
                pushes_from_owned_city = (
                    (city_focus_enabled or army_dominance)
                    and source_is_owned_city
                    and armies[source] > OWN_CITY_PUSH_ARMY
                    and armies[source] - 1 > OWN_CITY_ARMY_RESERVE
                    and (
                        target_is_new_tile
                        or follows_city_free_route
                        or follows_fallback_route
                        or is_closer
                    )
                )
                moves_toward_city = (
                    city_focus_enabled
                    and target_city_distance is not None
                    and (
                        source_city_distance is None
                        or target_city_distance < source_city_distance
                    )
                )
                chains_army_forward = (
                    (aggression_pulse or army_dominance)
                    and terrain[target] == self.state.player_index
                    and follows_city_free_route
                    and armies[source] >= AGGRESSION_MIN_CHAIN_ARMY
                )
                dominance_route_push = (
                    army_dominance
                    and armies[source] >= FLOW_CHAIN_MIN_ARMY
                    and (
                        follows_city_free_route
                        or follows_fallback_route
                        or is_closer
                    )
                )
                dominance_city_push = (
                    dominance_route_push
                    and source_is_owned_city
                    and armies[source] - 1 > OWN_CITY_ARMY_RESERVE
                )
                extends_flow_chain = (
                    armies[source] >= FLOW_CHAIN_MIN_ARMY
                    and self.extends_flow_chain(source, target)
                    and (
                        target_is_new_tile
                        or follows_city_free_route
                        or follows_fallback_route
                        or is_closer
                    )
                )
                drains_overstacked_tile = (
                    source != self.state.my_general_index
                    and source not in self.state.city_set()
                    and armies[source] > normal_tile_soft_cap
                    and (
                        target_is_new_tile
                        or follows_city_free_route
                        or follows_fallback_route
                        or is_closer
                    )
                )
                drains_overstacked_along_flow = drains_overstacked_tile and extends_flow_chain
                reinforcement_route_push = (
                    reinforcement.staging
                    and defense_target is None
                    and (
                        follows_city_free_route
                        or follows_fallback_route
                        or route_progress > 0
                        or fallback_route_progress > 0
                    )
                )
                reinforcement_frontier_push = (
                    reinforcement.expand_window
                    and defense_target is None
                    and target_is_new_tile
                    and not target_is_enemy_tile
                )
                reinforcement_attack_push = (
                    reinforcement.attack_window
                    and defense_target is None
                    and (
                        target_is_enemy_tile
                        or can_take_city
                        or attacks_known_general_route
                    )
                )
                reverses_recent_move = self.reverses_recent_move(source, target)
                repeats_recent_edge = self.repeats_recent_edge(source, target)
                repeats_recent_target = self.repeats_recent_target(target)
                stalemate_scout_push = (
                    stalemate_breakout
                    and target_is_new_tile
                    and not self.state.has_seen(target)
                )
                stalemate_route_push = (
                    stalemate_breakout
                    and (
                        follows_city_free_route
                        or follows_fallback_route
                        or route_progress > 0
                        or fallback_route_progress > 0
                    )
                )
                stalemate_repeat_move = (
                    stalemate_breakout
                    and (
                        reverses_recent_move
                        or repeats_recent_edge
                        or repeats_recent_target
                    )
                )
                stack_consolidation = self.is_stack_consolidation_move(
                    source,
                    target,
                    terrain,
                    armies,
                    source_distance,
                    target_distance,
                    route_progress,
                    fallback_route_progress,
                )
                oscillates_without_purpose = (
                    not defends_general
                    and not attacks_threat
                    and not target_is_enemy_general
                    and not target_is_enemy_tile
                    and not can_take_city
                    and not target_is_new_tile
                    and (
                        reverses_recent_move
                        or repeats_recent_edge
                        or repeats_recent_target
                    )
                )
                if oscillates_without_purpose:
                    continue
                if (
                    enemy_general_known
                    and defense_target is None
                    and not attacks_known_general_route
                    and not target_is_enemy_general
                ):
                    continue

                candidate = {
                    "source": source,
                    "target": target,
                    "target_is_new_tile": target_is_new_tile,
                    "target_is_enemy_tile": target_is_enemy_tile,
                    "target_is_enemy_general": target_is_enemy_general,
                    "target_is_priority_enemy_general": target_is_priority_enemy_general,
                    "can_take_city": can_take_city,
                    "follows_city_free_route": follows_city_free_route,
                    "follows_fallback_route": follows_fallback_route,
                    "route_progress": route_progress,
                    "fallback_route_progress": fallback_route_progress,
                    "opens_route_gateway": opens_route_gateway,
                    "broad_expansion": broad_expansion_mode and target_is_new_tile,
                    "frontier_score": self.frontier_score(target, terrain) if target_is_new_tile else 0,
                    "search_frontier_score": search_frontier_score,
                    "information_gain": information_gain,
                    "belief_proximity": belief_proximity,
                    "defends_general": defends_general,
                    "attacks_threat": attacks_threat,
                    "attacks_known_general_route": attacks_known_general_route,
                    "pushes_from_owned_city": pushes_from_owned_city,
                    "moves_toward_city": moves_toward_city,
                    "chains_army_forward": chains_army_forward,
                    "dominance_route_push": dominance_route_push,
                    "dominance_city_push": dominance_city_push,
                    "extends_flow_chain": extends_flow_chain,
                    "drains_overstacked_tile": drains_overstacked_tile,
                    "drains_overstacked_along_flow": drains_overstacked_along_flow,
                    "stack_consolidation": stack_consolidation,
                    "reinforcement_route_push": reinforcement_route_push,
                    "reinforcement_frontier_push": reinforcement_frontier_push,
                    "reinforcement_attack_push": reinforcement_attack_push,
                    "stalemate_breakout": stalemate_breakout,
                    "stalemate_scout_push": stalemate_scout_push,
                    "stalemate_route_push": stalemate_route_push,
                    "stalemate_repeat_move": stalemate_repeat_move,
                    "option": self.current_option.name,
                    "prediction_heat": self.state.enemy_prediction_heat.get(target, 0),
                    "reverses_recent_move": reverses_recent_move,
                    "repeats_recent_edge": repeats_recent_edge,
                    "repeats_recent_target": repeats_recent_target,
                    "half": (
                        False
                        if enemy_general_known
                        else self.should_half_move_from_general(source, armies[source], turn)
                    ),
                    "target_distance": target_distance,
                    "army_dominance": army_dominance,
                }

                utility = self.score_move_utility(candidate, armies)
                score = utility.score if utility is not None else None
                prioritized = self.priority_agent.prioritize(
                    candidate,
                    score,
                    self.recent_moves,
                    self.flow_chain,
                )
                if prioritized is not None:
                    candidate["base_score"] = score
                    candidate["score_components"] = utility.components
                    candidate["score"] = prioritized.score
                    candidate["priority_reason"] = prioritized.reason
                    moves.append(candidate)

        if not moves:
            self.log(
                "No move found:",
                "blocked tiles =", blocked_tiles,
                "ignored cities =", skipped_cities,
                "own movable tiles =", own_movable_tiles,
            )
            return None

        self.apply_lookahead_scores(moves, terrain, armies, strategy_target)
        self.apply_action_value_scores(moves, turn, terrain, armies)
        self.last_move_explanations = self.explain_top_moves(moves, armies)
        best_move = max(
            moves,
            key=lambda move: (
                move["score"],
                armies[move["source"]],
                -move["target_distance"],
            ),
        )
        source = best_move["source"]
        target = best_move["target"]
        half = best_move["half"]
        best_move["final_score"] = best_move["score"]
        self.last_selected_move = dict(best_move)
        self.remember_move(source, target)
        self.remember_flow(source, target)
        if best_move.get("priority_reason"):
            self.log(
                "Priority agent:",
                best_move["priority_reason"],
                "base =",
                best_move.get("base_score"),
                "final =",
                best_move["score"],
            )
        if best_move.get("lookahead_score"):
            self.log(
                "Lookahead:",
                "bonus =",
                best_move["lookahead_bonus"],
                "future =",
                best_move["lookahead_score"],
                "reason =",
                best_move.get("lookahead_reason"),
            )
        if best_move.get("action_value_adjustment"):
            self.log(
                "Action value:",
                "p =",
                best_move.get("action_value"),
                "adjust =",
                best_move.get("action_value_adjustment"),
                "samples =",
                best_move.get("action_value_sample_count"),
            )
        if best_move.get("stalemate_breakout"):
            self.log(
                "Stalemate breakout move:",
                source,
                "->",
                target,
                "reason =",
                self.stalemate.reason,
                "components =",
                {
                    key: value
                    for key, value in best_move.get("score_components", {}).items()
                    if key.startswith("stalemate")
                },
            )
        return Move(source, target, half, strategy_target)

    def explain_top_moves(self, moves, armies):
        top_moves = sorted(
            moves,
            key=lambda move: (
                move["score"],
                armies[move["source"]],
                -move["target_distance"],
            ),
            reverse=True,
        )[:MOVE_EXPLANATION_TOP_K]
        return [self.explain_move_candidate(move) for move in top_moves]

    def explain_move_candidate(self, move):
        components = move.get("score_components") or {}
        top_components = sorted(
            components.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:8]
        return {
            "source": move.get("source"),
            "target": move.get("target"),
            "score": move.get("score"),
            "base_score": move.get("base_score"),
            "target_distance": move.get("target_distance"),
            "option": move.get("option"),
            "priority_reason": move.get("priority_reason"),
            "lookahead_bonus": move.get("lookahead_bonus", 0),
            "lookahead_reason": move.get("lookahead_reason"),
            "action_value": move.get("action_value"),
            "action_value_adjustment": move.get("action_value_adjustment", 0),
            "top_components": dict(top_components),
            "flags": {
                key: bool(move.get(key))
                for key in (
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
            },
        }

    def apply_action_value_scores(self, moves, turn, terrain, armies):
        if not moves or self.action_value_agent is None:
            return
        if not self.action_value_agent.is_ready():
            return

        beam = sorted(
            moves,
            key=lambda move: (
                move["score"],
                armies[move["source"]],
                -move["target_distance"],
            ),
            reverse=True,
        )[:ACTION_VALUE_BEAM_WIDTH]

        for move in beam:
            sample = self.action_value_sample(move, turn, terrain, armies)
            result = self.action_value_agent.score_adjustment(
                sample,
                score_cap=ACTION_VALUE_SCORE_CAP,
            )
            if not result:
                continue
            adjustment = result["score_adjustment"]
            move["action_value"] = result["action_value"]
            move["action_value_adjustment"] = adjustment
            move["action_value_sample_count"] = result["sample_count"]
            move.setdefault("score_components", {})["action_value"] = adjustment
            move["score"] += adjustment

    def action_value_sample(self, move, turn, terrain, armies):
        source = move["source"]
        target = move["target"]
        return {
            "move_number": len(self.recent_moves) + 1,
            "turn": turn,
            "visible_turn": self.state.visible_turn(turn),
            "source": source,
            "target": target,
            "half": move.get("half", False),
            "strategy_target": self.planned_target,
            "source_army": armies[source] if source < len(armies) else 0,
            "target_army": armies[target] if target < len(armies) else 0,
            "target_terrain": terrain[target] if target < len(terrain) else None,
            "target_distance": move.get("target_distance"),
            "my_tiles": self.state.my_tile_count(),
            "enemy_tiles": self.state.biggest_enemy_tile_count(),
            "my_army": self.state.my_total_army(),
            "enemy_army": self.state.biggest_enemy_total_army(),
            "visible_enemy_tiles": len(self.state.visible_enemy_tiles),
            "seen_tiles": len(self.state.seen_tiles),
            "width": self.state.width,
            "height": self.state.height,
            "score": move.get("score"),
            "base_score": move.get("base_score"),
            "lookahead_bonus": move.get("lookahead_bonus", 0),
            "score_components": move.get("score_components") or {},
            "flags": {
                key: bool(move.get(key))
                for key in (
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
            },
        }

    def apply_lookahead_scores(self, moves, terrain, armies, strategy_target):
        if not moves or LOOKAHEAD_BEAM_WIDTH <= 0 or LOOKAHEAD_SCORE_WEIGHT <= 0:
            return

        beam = sorted(
            moves,
            key=lambda move: (
                move["score"],
                armies[move["source"]],
                -move["target_distance"],
            ),
            reverse=True,
        )[:LOOKAHEAD_BEAM_WIDTH]

        for move in beam:
            score, reason = self.lookahead_score(move, terrain, armies, strategy_target)
            bonus = int(min(LOOKAHEAD_SCORE_CAP, score) * LOOKAHEAD_SCORE_WEIGHT)
            if bonus <= 0:
                continue
            move["lookahead_score"] = score
            move["lookahead_bonus"] = bonus
            move["lookahead_reason"] = reason
            move.setdefault("score_components", {})["lookahead"] = bonus
            move["score"] += bonus

    def lookahead_score(self, move, terrain, armies, strategy_target):
        virtual_terrain, virtual_armies = self.virtual_after_move(move, terrain, armies)
        if virtual_terrain is None:
            return 0, None

        local_sources = self.lookahead_sources(move, virtual_terrain, virtual_armies)
        best_score = 0
        best_reason = None

        for source in local_sources:
            source_distance = self.route_distance(source, strategy_target)
            for target in neighbors(self.state, source):
                score, reason = self.lookahead_transition_score(
                    source,
                    target,
                    source_distance,
                    virtual_terrain,
                    virtual_armies,
                    strategy_target,
                    move["source"],
                )
                if score > best_score:
                    best_score = score
                    best_reason = reason

        return best_score, best_reason

    def virtual_after_move(self, move, terrain, armies):
        source = move["source"]
        target = move["target"]
        if source >= len(terrain) or target >= len(terrain):
            return None, None

        virtual_terrain = list(terrain)
        virtual_armies = list(armies)
        source_army = max(0, virtual_armies[source])
        if source_army <= 1:
            return None, None

        moved_army = source_army // 2 if move.get("half") else source_army - 1
        if moved_army <= 0:
            return None, None

        virtual_armies[source] = source_army - moved_army
        if virtual_terrain[target] == self.state.player_index:
            virtual_armies[target] += moved_army
            return virtual_terrain, virtual_armies

        target_army = max(0, virtual_armies[target])
        if moved_army > target_army:
            virtual_terrain[target] = self.state.player_index
            virtual_armies[target] = moved_army - target_army
        else:
            virtual_armies[target] = target_army - moved_army

        return virtual_terrain, virtual_armies

    def lookahead_sources(self, move, terrain, armies):
        candidates = {move["target"]}
        for index in neighbors(self.state, move["target"]):
            candidates.add(index)

        return [
            index
            for index in candidates
            if (
                0 <= index < len(terrain)
                and terrain[index] == self.state.player_index
                and armies[index] > 1
            )
        ]

    def route_distance(self, index, strategy_target):
        if strategy_target is None:
            return None

        distance = self.planned_distances.get(index)
        if distance is not None:
            return distance
        distance = self.planned_fallback_distances.get(index)
        if distance is not None:
            return distance
        return distance_to_target(self.state, index, strategy_target)

    def lookahead_transition_score(
        self,
        source,
        target,
        source_distance,
        terrain,
        armies,
        strategy_target,
        previous_source,
    ):
        if target >= len(terrain) or not is_passable(terrain[target]):
            return 0, None
        if terrain[target] >= 0 and terrain[target] != self.state.player_index:
            if armies[source] <= armies[target] + 1:
                return 0, None

        score = 0
        reasons = []
        target_distance = self.route_distance(target, strategy_target)
        if source_distance is not None and target_distance is not None:
            progress = source_distance - target_distance
            if progress > 0:
                score += min(9000, progress * 4500)
                reasons.append(f"route +{progress}")

        if terrain[target] != self.state.player_index:
            score += 4500
            reasons.append("future expansion")
        if terrain[target] >= 0 and terrain[target] != self.state.player_index:
            score += 6500
            reasons.append("future attack")
        if target in self.state.city_set():
            score += 5000
            reasons.append("future city")
        if target == self.state.enemy_general_index:
            score += LOOKAHEAD_SCORE_CAP
            reasons.append("future general")
        if self.search_frontier_score(target, terrain, strategy_target) > 0:
            score += 2500
            reasons.append("future scout")
        if target == previous_source:
            score -= 7000
            reasons.append("pullback")

        if score <= 0:
            return 0, None
        return score, ", ".join(reasons[:3])

    def score_move(self, move, armies):
        utility = self.score_move_utility(move, armies)
        return utility.score if utility is not None else None

    def score_move_utility(self, move, armies):
        has_purpose = (
            move["defends_general"]
            or move["attacks_threat"]
            or move["target_is_enemy_general"]
            or move["target_is_enemy_tile"]
            or move["can_take_city"]
            or move["target_is_new_tile"]
        )
        if (move["reverses_recent_move"] or move["repeats_recent_edge"]) and not has_purpose:
            return None
        if move["repeats_recent_target"] and not has_purpose:
            return None

        components = {}

        def add(component, value):
            if not value:
                return
            components[component] = components.get(component, 0) + int(value)

        if move["attacks_threat"]:
            add("attack_threat", 100000)
        if move["defends_general"]:
            add("defend_general", 90000)
        if move["target_is_enemy_general"]:
            add("known_general_target", KNOWN_GENERAL_TARGET_BONUS)
        if move["attacks_known_general_route"]:
            add("known_general_route", KNOWN_GENERAL_ROUTE_BONUS)
        if move["target_is_priority_enemy_general"]:
            add("priority_enemy_general", 120000 if move["army_dominance"] else 80000)
        elif move["target_is_enemy_general"]:
            add("enemy_general", 90000 if move["army_dominance"] else 50000)
        if move["can_take_city"] and (
            not move["attacks_known_general_route"]
            or move["follows_city_free_route"]
            or move["follows_fallback_route"]
        ):
            add("take_city", 32000)
        if move["target_is_enemy_tile"]:
            add("attack_enemy_tile", 56000 if move["army_dominance"] else 30000)

        attack_bias = self.coach_bias.get("attack_bias", 0.0)
        defense_bias = self.coach_bias.get("defense_bias", 0.0)
        expansion_bias = self.coach_bias.get("expansion_bias", 0.0)
        city_bias = self.coach_bias.get("city_bias", 0.0)
        route_bias = self.coach_bias.get("route_bias", 0.0)

        if move["target_is_enemy_tile"] or move["attacks_known_general_route"]:
            add("coach_attack_bias", attack_bias * 5000)
        if move["target_is_enemy_general"]:
            add("coach_general_attack_bias", attack_bias * 15000)
        if move["defends_general"] or move["attacks_threat"]:
            add("coach_defense_bias", defense_bias * 12000)
        if move["target_is_new_tile"]:
            add("coach_expansion_bias", expansion_bias * 3500)
        if move["broad_expansion"]:
            add("coach_broad_expansion_bias", expansion_bias * 2500)
        if move["can_take_city"] or move["moves_toward_city"] or move["pushes_from_owned_city"]:
            add("coach_city_bias", city_bias * 3500)
        if (
            move["follows_city_free_route"]
            or move["follows_fallback_route"]
            or move["route_progress"] > 0
            or move["fallback_route_progress"] > 0
        ):
            add("coach_route_bias", route_bias * 2500)

        if move["broad_expansion"]:
            add("broad_expansion", 13000)
        if move["target_is_new_tile"]:
            add("new_tile", 14000)
        if move["follows_city_free_route"]:
            add("city_free_route", 4500)
        if move["follows_fallback_route"]:
            add("fallback_route", 2500)
        if move["route_progress"] > 0:
            add("route_progress", move["route_progress"] * ROUTE_PROGRESS_BONUS)
        if move["fallback_route_progress"] > 0:
            add("fallback_route_progress", move["fallback_route_progress"] * FALLBACK_ROUTE_PROGRESS_BONUS)
        if move["opens_route_gateway"]:
            add("route_gateway", ROUTE_GATEWAY_BONUS)

        if move["pushes_from_owned_city"]:
            add("owned_city_push", 5000)
        if move["moves_toward_city"]:
            add("toward_city", 3000)
        if move["chains_army_forward"]:
            add("army_chain", 4200)
        if move["dominance_route_push"]:
            add("dominance_route_push", DOMINANCE_ROUTE_PUSH_BONUS)
        if move["dominance_city_push"]:
            add("dominance_city_push", DOMINANCE_CITY_PUSH_BONUS)
        if move["extends_flow_chain"]:
            add("flow_chain", 3600)
        if move["drains_overstacked_along_flow"]:
            add("drain_overstacked_flow", 3200)
        elif move["drains_overstacked_tile"]:
            add("drain_overstacked", 1800)
        if move["stack_consolidation"]:
            add("stack_consolidation", STACK_CONSOLIDATION_BONUS)
        if move["reinforcement_route_push"]:
            add("reinforcement_route", REINFORCEMENT_ROUTE_BONUS)
        if move["reinforcement_frontier_push"]:
            add("reinforcement_frontier", REINFORCEMENT_FRONTIER_BONUS)
        if move["reinforcement_attack_push"]:
            add("reinforcement_attack", REINFORCEMENT_ATTACK_BONUS)
        if move.get("stalemate_scout_push"):
            add("stalemate_scout", STALEMATE_SCOUT_BONUS)
        if move.get("stalemate_breakout") and move["target_is_new_tile"]:
            add("stalemate_new_tile", STALEMATE_NEW_TILE_BONUS)
        if move.get("stalemate_route_push"):
            add("stalemate_route", STALEMATE_ROUTE_BONUS)
        if move.get("stalemate_breakout") and move["can_take_city"]:
            add("stalemate_city", STALEMATE_CITY_BONUS)
        if move.get("stalemate_repeat_move"):
            add("stalemate_repeat_penalty", -STALEMATE_REPEAT_PENALTY)
        option = move.get("option")
        if option == "secure_general" and (move["defends_general"] or move["attacks_threat"]):
            add("option_secure_general", 12000)
        if option == "attack_general" and (move["target_is_enemy_general"] or move["attacks_known_general_route"]):
            add("option_attack_general", 14000)
        if option == "scout_general" and (move["target_is_new_tile"] or move["search_frontier_score"] > 0):
            add("option_scout_general", 9000)
        if option == "contest_city" and (move["can_take_city"] or move["moves_toward_city"]):
            add("option_contest_city", 8000)
        if option == "break_stalemate" and (move.get("stalemate_scout_push") or move["target_is_new_tile"]):
            add("option_break_stalemate", 10000)
        if option == "press_route" and (
            move["follows_city_free_route"]
            or move["follows_fallback_route"]
            or move["route_progress"] > 0
        ):
            add("option_press_route", 6000)
        if move["defends_general"] and move["prediction_heat"]:
            add("enemy_prediction_heat", move["prediction_heat"] * 1200)

        add("frontier", move["frontier_score"] * 300)
        search_multiplier = 700 if move["army_dominance"] else 250
        add("search_frontier", move["search_frontier_score"] * search_multiplier)
        add("information_gain", move.get("information_gain", 0) * INFORMATION_GAIN_BONUS)
        add(
            "belief_proximity",
            move.get("belief_proximity", 0) * INFORMATION_BELIEF_PROXIMITY_BONUS // 10,
        )
        add("target_distance", -move["target_distance"] * ROUTE_DISTANCE_PENALTY)

        if move["reverses_recent_move"]:
            add("reverse_penalty", -20000)
        if move["repeats_recent_edge"]:
            add("edge_repeat_penalty", -15000)
        if move["repeats_recent_target"]:
            add("target_repeat_penalty", -12000)

        add("source_army", min(armies[move["source"]], 80))
        return MoveUtility(score=sum(components.values()), components=components)

    def is_stack_consolidation_move(
        self,
        source,
        target,
        terrain,
        armies,
        source_distance,
        target_distance,
        route_progress,
        fallback_route_progress,
    ):
        if terrain[target] != self.state.player_index:
            return False
        if source == self.state.my_general_index:
            return False
        if target in self.state.city_set() and armies[target] >= armies[source]:
            return False
        if armies[source] + armies[target] < STACK_CONSOLIDATION_MIN_ARMY:
            return False
        if not self.has_nearby_stack_partner(source, target, terrain, armies):
            return False

        moves_toward_route = route_progress > 0 or fallback_route_progress > 0
        if moves_toward_route:
            return True

        return target_distance <= source_distance

    def has_nearby_stack_partner(self, source, target, terrain, armies):
        for index, owner in enumerate(terrain):
            if index in (source, target):
                continue
            if owner != self.state.player_index:
                continue
            if armies[index] < STACK_CONSOLIDATION_MIN_ARMY:
                continue
            if distance_to_target(self.state, source, index) <= STACK_CONSOLIDATION_RADIUS:
                return True
            if distance_to_target(self.state, target, index) <= STACK_CONSOLIDATION_RADIUS:
                return True
        return False

    def reverses_recent_move(self, source, target):
        return (target, source) in self.recent_moves[-8:]

    def repeats_recent_edge(self, source, target):
        edge = frozenset((source, target))
        return sum(
            1
            for recent_source, recent_target in self.recent_moves[-LOOP_DETECTION_MEMORY:]
            if frozenset((recent_source, recent_target)) == edge
        ) >= LOOP_TARGET_VISIT_LIMIT

    def repeats_recent_target(self, target):
        return sum(
            1
            for _, recent_target in self.recent_moves[-LOOP_DETECTION_MEMORY:]
            if recent_target == target
        ) >= LOOP_TARGET_VISIT_LIMIT

    def remember_move(self, source, target):
        self.recent_moves.append((source, target))
        if len(self.recent_moves) > LOOP_DETECTION_MEMORY:
            self.recent_moves = self.recent_moves[-LOOP_DETECTION_MEMORY:]

    def extends_flow_chain(self, source, target):
        if not self.flow_chain:
            return False

        last_source, last_target = self.flow_chain[-1]
        if source == last_target:
            return True

        return any(
            source == recent_target and target != recent_source
            for recent_source, recent_target in self.flow_chain[-FLOW_CHAIN_MEMORY:]
        )

    def remember_flow(self, source, target):
        if self.extends_flow_chain(source, target):
            self.flow_chain.append((source, target))
        else:
            self.flow_chain = [(source, target)]

        if len(self.flow_chain) > FLOW_CHAIN_MEMORY:
            self.flow_chain = self.flow_chain[-FLOW_CHAIN_MEMORY:]
