from datetime import datetime

from action_encoding import direction_between
from config import (
    CITY_FOCUS_TURN_MIN,
    GENERAL_ATTACK_TURN_MIN,
    RESERVE_TURN_MIN,
    TILE_FOG,
    TILE_FOG_OBSTACLE,
    TILE_MOUNTAIN,
)
from game_state import GameState
from strategy import Strategy
from strategy_coach import StrategyCoach


class SimStrategyAgent:
    def __init__(self, player_index=0, label="sim-agent"):
        self.player_index = player_index
        self.enemy_player_index = 1 - player_index
        self.label = label
        self.state = GameState(player_index=player_index, enemy_player_index=self.enemy_player_index)
        self.strategy = Strategy(
            self.state,
            reserve_after_turn=RESERVE_TURN_MIN,
            city_focus_after_turn=CITY_FOCUS_TURN_MIN,
            general_attack_after_turn=GENERAL_ATTACK_TURN_MIN,
        )
        self.coach = StrategyCoach(path="data/profiles/sim_strategy_profile.json")
        self.coach.apply_start_profile(self.strategy)
        self.records = []
        self.move_count = 0

    def act(self, observation, key):
        import jax.numpy as jnp

        self.update_state(observation)
        self.coach.observe(self.state, self.strategy, self.state.turn)
        move = self.strategy.choose_move(self.state.turn)
        if move is None:
            return jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

        direction = direction_between(self.state.width, move.source, move.target)
        if direction is None:
            return jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

        self.move_count += 1
        self.record_move(move)
        row = move.source // self.state.width
        col = move.source % self.state.width
        return jnp.array([0, row, col, direction, int(move.half)], dtype=jnp.int32)

    def update_state(self, observation):
        import numpy as np

        armies_grid = np.asarray(observation.armies).astype(int)
        height, width = armies_grid.shape
        tile_count = width * height
        armies = armies_grid.reshape(-1).tolist()
        terrain = [TILE_FOG] * tile_count
        cities = []
        generals = [-1, -1]

        owned = np.asarray(observation.owned_cells).astype(bool).reshape(-1)
        opponent = np.asarray(observation.opponent_cells).astype(bool).reshape(-1)
        neutral = np.asarray(observation.neutral_cells).astype(bool).reshape(-1)
        mountains = np.asarray(observation.mountains).astype(bool).reshape(-1)
        fog = np.asarray(observation.fog_cells).astype(bool).reshape(-1)
        structures_in_fog = np.asarray(observation.structures_in_fog).astype(bool).reshape(-1)
        city_mask = np.asarray(observation.cities).astype(bool).reshape(-1)
        general_mask = np.asarray(observation.generals).astype(bool).reshape(-1)

        for index in range(tile_count):
            if owned[index]:
                terrain[index] = self.player_index
            elif opponent[index]:
                terrain[index] = self.enemy_player_index
            elif neutral[index]:
                terrain[index] = -1
            elif mountains[index]:
                terrain[index] = TILE_MOUNTAIN
            elif structures_in_fog[index]:
                terrain[index] = TILE_FOG_OBSTACLE
            elif fog[index]:
                terrain[index] = TILE_FOG

            if city_mask[index]:
                cities.append(index)
            if general_mask[index] and owned[index]:
                generals[self.player_index] = index
            elif general_mask[index] and opponent[index]:
                generals[self.enemy_player_index] = index

        self.state.update(
            {
                "map": [width, height] + armies + terrain,
                "cities": cities,
                "generals": generals,
                "scores": [
                    {
                        "i": self.player_index,
                        "tiles": int(observation.owned_land_count),
                        "total": int(observation.owned_army_count),
                    },
                    {
                        "i": self.enemy_player_index,
                        "tiles": int(observation.opponent_land_count),
                        "total": int(observation.opponent_army_count),
                    },
                ],
                "turn": int(observation.timestep),
            }
        )

    def record_move(self, move):
        selected = self.strategy.last_selected_move or {}
        self.records.append(
            {
                "source": move.source,
                "target": move.target,
                "half": move.half,
                "move_number": self.move_count,
                "turn": self.state.turn,
                "visible_turn": self.state.visible_turn(self.state.turn),
                "width": self.state.width,
                "height": self.state.height,
                "my_tiles": self.state.my_tile_count(),
                "enemy_tiles": self.state.biggest_enemy_tile_count(),
                "my_army": self.state.my_total_army(),
                "enemy_army": self.state.biggest_enemy_total_army(),
                "score": selected.get("final_score", selected.get("score")),
                "base_score": selected.get("base_score"),
                "target_distance": selected.get("target_distance"),
                "score_components": selected.get("score_components") or {},
                "flags": {
                    key: bool(selected.get(key))
                    for key in (
                        "target_is_new_tile",
                        "target_is_enemy_tile",
                        "target_is_enemy_general",
                        "can_take_city",
                        "defends_general",
                        "attacks_threat",
                    )
                },
            }
        )

    def finished_records(self, won, opponent, final_info):
        result = []
        for record in self.records:
            result.append(
                {
                    **record,
                    "run_id": self.label,
                    "data_source": "simulator",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "opponent": opponent,
                    "won": won,
                    "reward": 1 if won is True else -1 if won is False else 0,
                    "final_turn": int(final_info.time) if final_info is not None else None,
                }
            )
        self.records = []
        return result
