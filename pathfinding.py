from collections import deque

from config import TILE_FOG_OBSTACLE, TILE_MOUNTAIN


def index_to_xy(state, index):
    return index % state.width, index // state.width


def xy_to_index(state, x, y):
    return y * state.width + x


def neighbors(state, index):
    x, y = index_to_xy(state, index)

    if y > 0:
        yield xy_to_index(state, x, y - 1)
    if x < state.width - 1:
        yield xy_to_index(state, x + 1, y)
    if y < state.height - 1:
        yield xy_to_index(state, x, y + 1)
    if x > 0:
        yield xy_to_index(state, x - 1, y)


def is_passable(tile_value):
    return tile_value not in (TILE_MOUNTAIN, TILE_FOG_OBSTACLE)


def distance_to_target(state, index, target):
    x, y = index_to_xy(state, index)
    target_x, target_y = index_to_xy(state, target)
    return abs(x - target_x) + abs(y - target_y)


def distance_to_center(state, index):
    x, y = index_to_xy(state, index)
    center_x = (state.width - 1) / 2
    center_y = (state.height - 1) / 2
    return abs(x - center_x) + abs(y - center_y)


def reachable_tiles_from(state, start, terrain, avoid_cities=True):
    if start is None or len(terrain) < state.width * state.height:
        return set()

    blocked_cities = state.city_set() if avoid_cities else set()
    queue = deque([start])
    reachable = {start}

    while queue:
        current = queue.popleft()

        for next_index in neighbors(state, current):
            if next_index in reachable or not is_passable(terrain[next_index]):
                continue
            if next_index in blocked_cities:
                continue

            reachable.add(next_index)
            queue.append(next_index)

    return reachable


def build_distance_map(state, target, terrain, avoid_cities=True):
    if target is None or state.width == 0 or state.height == 0:
        return {}

    if len(terrain) < state.width * state.height:
        return {}

    blocked_cities = state.city_set() if avoid_cities else set()
    queue = deque([target])
    distances = {target: 0}

    while queue:
        current = queue.popleft()

        for next_index in neighbors(state, current):
            if next_index in distances or not is_passable(terrain[next_index]):
                continue
            if next_index in blocked_cities and next_index != target:
                continue

            distances[next_index] = distances[current] + 1
            queue.append(next_index)

    return distances
