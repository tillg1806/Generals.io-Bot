from config import TILE_FOG_OBSTACLE, TILE_MOUNTAIN
from decoded_action import DecodedAction


DIRECTIONS = (
    (0, -1),  # up
    (0, 1),   # down
    (-1, 0),  # left
    (1, 0),   # right
)
FULL_MOVE_PLANES = 4
HALF_MOVE_OFFSET = 4
PASS_PLANE = 8
ACTION_PLANES = 9

def action_count(width, height):
    return ACTION_PLANES * max(0, width) * max(0, height)


def direction_between(width, source, target):
    if width <= 0 or source is None or target is None:
        return None

    source_x = source % width
    source_y = source // width
    target_x = target % width
    target_y = target // width
    delta = (target_x - source_x, target_y - source_y)
    try:
        return DIRECTIONS.index(delta)
    except ValueError:
        return None


def encode_action(width, height, source=None, target=None, half=False, is_pass=False):
    tile_count = max(0, width) * max(0, height)
    if tile_count <= 0:
        return None

    if is_pass:
        return PASS_PLANE * tile_count

    if source is None or target is None:
        return None
    if source < 0 or source >= tile_count or target < 0 or target >= tile_count:
        return None

    direction = direction_between(width, source, target)
    if direction is None:
        return None

    plane = direction + (HALF_MOVE_OFFSET if half else 0)
    return plane * tile_count + source


def decode_action(width, height, action_index):
    tile_count = max(0, width) * max(0, height)
    if tile_count <= 0 or action_index is None:
        return None
    if action_index < 0 or action_index >= ACTION_PLANES * tile_count:
        return None

    plane = action_index // tile_count
    source = action_index % tile_count
    if plane == PASS_PLANE:
        return DecodedAction(
            source=None,
            target=None,
            half=False,
            is_pass=True,
            direction=None,
        )
    if plane > PASS_PLANE:
        return None

    direction = plane % FULL_MOVE_PLANES
    half = plane >= HALF_MOVE_OFFSET
    source_x = source % width
    source_y = source // width
    dx, dy = DIRECTIONS[direction]
    target_x = source_x + dx
    target_y = source_y + dy
    if target_x < 0 or target_x >= width or target_y < 0 or target_y >= height:
        return None

    target = target_y * width + target_x
    return DecodedAction(
        source=source,
        target=target,
        half=half,
        is_pass=False,
        direction=direction,
    )


def is_blocked_tile(tile):
    return tile in (TILE_MOUNTAIN, TILE_FOG_OBSTACLE)


def legal_action_mask(state, terrain=None, armies=None, allow_pass=True):
    terrain, armies = _terrain_and_armies(state, terrain, armies)
    tile_count = state.width * state.height
    mask = [0] * action_count(state.width, state.height)
    if tile_count <= 0 or len(terrain) < tile_count or len(armies) < tile_count:
        return mask

    if allow_pass:
        mask[PASS_PLANE * tile_count] = 1

    for source, owner in enumerate(terrain[:tile_count]):
        if owner != state.player_index or armies[source] <= 1:
            continue

        source_x = source % state.width
        source_y = source // state.width
        for direction, (dx, dy) in enumerate(DIRECTIONS):
            target_x = source_x + dx
            target_y = source_y + dy
            if target_x < 0 or target_x >= state.width or target_y < 0 or target_y >= state.height:
                continue

            target = target_y * state.width + target_x
            if is_blocked_tile(terrain[target]):
                continue

            mask[direction * tile_count + source] = 1
            mask[(HALF_MOVE_OFFSET + direction) * tile_count + source] = 1

    return mask


def legal_action_planes(state, terrain=None, armies=None, allow_pass=True):
    flat = legal_action_mask(state, terrain=terrain, armies=armies, allow_pass=allow_pass)
    tile_count = state.width * state.height
    if tile_count <= 0:
        return []

    return [
        flat[plane * tile_count:(plane + 1) * tile_count]
        for plane in range(ACTION_PLANES)
    ]


def _terrain_and_armies(state, terrain, armies):
    if terrain is not None and armies is not None:
        return terrain, armies
    return state.split_map()
