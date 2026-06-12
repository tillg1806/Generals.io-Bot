from config import TILE_FOG, TILE_FOG_OBSTACLE, TILE_MOUNTAIN


BOARD_CHANNELS = (
    "army",
    "own_army",
    "enemy_army",
    "neutral_army",
    "own_tile",
    "enemy_tile",
    "neutral_tile",
    "fog",
    "fog_obstacle",
    "mountain",
    "city",
    "own_general",
    "known_enemy_general",
    "seen",
    "last_seen_age",
    "enemy_movement_heat",
    "enemy_prediction_heat",
    "x",
    "y",
    "turn",
    "own_land",
    "enemy_land",
    "own_total_army",
    "enemy_total_army",
)


def build_board_tensor(state, terrain=None, armies=None):
    terrain, armies = _terrain_and_armies(state, terrain, armies)
    tile_count = state.width * state.height
    if tile_count <= 0 or len(terrain) < tile_count or len(armies) < tile_count:
        return {
            "version": 1,
            "width": state.width,
            "height": state.height,
            "channels": list(BOARD_CHANNELS),
            "data": [],
        }

    channels = [[] for _ in BOARD_CHANNELS]
    max_x = max(1, state.width - 1)
    max_y = max(1, state.height - 1)
    own_land = _score_value(state, "tiles", state.player_index)
    enemy_land = state.biggest_enemy_tile_count()
    own_total_army = state.my_total_army()
    enemy_total_army = state.biggest_enemy_total_army()
    visible_turn = state.visible_turn(state.turn)

    for index in range(tile_count):
        owner = terrain[index]
        army = armies[index]
        is_own = owner == state.player_index
        is_enemy = owner >= 0 and owner != state.player_index
        is_neutral = owner == -1
        x = index % state.width
        y = index // state.width

        values = {
            "army": _army_norm(army),
            "own_army": _army_norm(army) if is_own else 0.0,
            "enemy_army": _army_norm(army) if is_enemy else 0.0,
            "neutral_army": _army_norm(army) if is_neutral else 0.0,
            "own_tile": 1.0 if is_own else 0.0,
            "enemy_tile": 1.0 if is_enemy else 0.0,
            "neutral_tile": 1.0 if is_neutral else 0.0,
            "fog": 1.0 if owner == TILE_FOG else 0.0,
            "fog_obstacle": 1.0 if owner == TILE_FOG_OBSTACLE else 0.0,
            "mountain": 1.0 if owner == TILE_MOUNTAIN else 0.0,
            "city": 1.0 if index in state.city_set() else 0.0,
            "own_general": 1.0 if index == state.my_general_index else 0.0,
            "known_enemy_general": 1.0 if index == state.enemy_general_index else 0.0,
            "seen": 1.0 if index in state.seen_tiles else 0.0,
            "last_seen_age": _age_norm(state.turn - state.last_seen(index))
            if state.last_seen(index) >= 0
            else 1.0,
            "enemy_movement_heat": _heat_norm(state.enemy_movement_heat.get(index, 0)),
            "enemy_prediction_heat": _heat_norm(state.enemy_prediction_heat.get(index, 0)),
            "x": x / max_x,
            "y": y / max_y,
            "turn": _turn_norm(visible_turn),
            "own_land": _land_norm(own_land, tile_count),
            "enemy_land": _land_norm(enemy_land, tile_count),
            "own_total_army": _army_norm(own_total_army),
            "enemy_total_army": _army_norm(enemy_total_army),
        }

        for channel_index, name in enumerate(BOARD_CHANNELS):
            channels[channel_index].append(round(float(values[name]), 6))

    return {
        "version": 1,
        "width": state.width,
        "height": state.height,
        "channels": list(BOARD_CHANNELS),
        "data": channels,
    }


def build_board_snapshot(state, terrain=None, armies=None, include_tensor=True):
    terrain, armies = _terrain_and_armies(state, terrain, armies)
    snapshot = {
        "version": 1,
        "width": state.width,
        "height": state.height,
        "turn": state.turn,
        "visible_turn": state.visible_turn(state.turn),
        "player_index": state.player_index,
        "my_general_index": state.my_general_index,
        "enemy_general_index": state.enemy_general_index,
        "city_count": len(state.cities),
        "seen_tile_count": len(state.seen_tiles),
        "visible_enemy_tile_count": len(state.visible_enemy_tiles),
        "channels": list(BOARD_CHANNELS),
    }
    if include_tensor:
        snapshot["tensor"] = build_board_tensor(state, terrain=terrain, armies=armies)["data"]
    return snapshot


def _terrain_and_armies(state, terrain, armies):
    if terrain is not None and armies is not None:
        return terrain, armies
    return state.split_map()


def _army_norm(value):
    return min(max(float(value or 0), 0.0), 1000.0) / 1000.0


def _heat_norm(value):
    return min(max(float(value or 0), 0.0), 20.0) / 20.0


def _age_norm(value):
    return min(max(float(value or 0), 0.0), 200.0) / 200.0


def _turn_norm(value):
    return min(max(float(value or 0), 0.0), 500.0) / 500.0


def _land_norm(value, tile_count):
    return min(max(float(value or 0), 0.0), float(tile_count)) / max(1.0, float(tile_count))


def _score_value(state, key, player_index):
    if player_index is None:
        return 0
    for score in state.scores:
        if score.get("i") == player_index:
            return score.get(key) or 0
    return 0
