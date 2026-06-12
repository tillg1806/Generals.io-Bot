from dataclasses import dataclass


@dataclass
class CoachSnapshot:
    turn: int
    visible_turn: int
    my_tiles: int
    enemy_tiles: int
    my_army: int
    enemy_army: int
    visible_enemy_tiles: int
    my_army_delta: int = 0
    enemy_army_delta: int = 0
    my_tile_delta: int = 0
    enemy_tile_delta: int = 0
    visible_my_cities: int = 0
    visible_enemy_cities: int = 0
    suspected_enemy_city_advantage: bool = False
