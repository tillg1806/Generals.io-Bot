from collections import deque

from config import TILE_FOG, TILE_FOG_OBSTACLE, TILE_MOUNTAIN
from pathfinding import build_distance_map, distance_to_target, is_passable, xy_to_index


def analyze_state_map(state, map_data=None, status=None):
    if state.width <= 0 or state.height <= 0:
        return None

    if map_data is None:
        map_data = state.map_data

    tile_count = state.width * state.height
    if len(map_data) < 2 + tile_count * 2:
        return None

    armies = map_data[2:2 + tile_count]
    terrain = map_data[2 + tile_count:2 + tile_count * 2]
    city_set = {city for city in state.city_set() if 0 <= city < tile_count}

    analysis = {
        "analysis_version": 1,
        "status": status,
        "width": state.width,
        "height": state.height,
        "tile_count": tile_count,
        "my_general_index": state.my_general_index,
        "enemy_general_index": state.enemy_general_index,
        "visibility": visibility_metrics(terrain),
        "terrain": terrain_metrics(terrain),
        "cities": city_metrics(state, terrain, city_set),
        "spawn": spawn_metrics(state, terrain),
        "symmetry": symmetry_metrics(state, terrain, city_set),
        "regions": region_metrics(state, terrain),
        "mountains": mountain_metrics(state, terrain),
        "quadrants": quadrant_metrics(state, terrain, city_set),
        "armies": army_metrics(armies, terrain),
    }
    return analysis


def visibility_metrics(terrain):
    tile_count = len(terrain)
    fog_count = sum(1 for tile in terrain if tile == TILE_FOG)
    fog_obstacle_count = sum(1 for tile in terrain if tile == TILE_FOG_OBSTACLE)
    visible_count = tile_count - fog_count - fog_obstacle_count
    return {
        "fog_count": fog_count,
        "fog_obstacle_count": fog_obstacle_count,
        "visible_tile_count": visible_count,
        "visible_ratio": round(visible_count / tile_count, 4) if tile_count else 0,
        "is_full_map_visible": fog_count == 0 and fog_obstacle_count == 0,
    }


def terrain_metrics(terrain):
    tile_count = len(terrain)
    mountain_count = sum(1 for tile in terrain if tile == TILE_MOUNTAIN)
    empty_count = sum(1 for tile in terrain if tile == -1)
    owned_count = sum(1 for tile in terrain if tile >= 0)
    passable_count = sum(1 for tile in terrain if is_passable(tile))
    return {
        "empty_count": empty_count,
        "mountain_count": mountain_count,
        "mountain_density": round(mountain_count / tile_count, 4) if tile_count else 0,
        "owned_count": owned_count,
        "passable_count": passable_count,
        "passable_density": round(passable_count / tile_count, 4) if tile_count else 0,
    }


def city_metrics(state, terrain, city_set):
    tile_count = len(terrain)
    city_distances_from_me = []
    city_distances_from_enemy = []

    for city in sorted(city_set):
        if state.my_general_index is not None:
            city_distances_from_me.append(distance_to_target(state, state.my_general_index, city))
        if state.enemy_general_index is not None:
            city_distances_from_enemy.append(distance_to_target(state, state.enemy_general_index, city))

    return {
        "city_count": len(city_set),
        "city_density": round(len(city_set) / tile_count, 4) if tile_count else 0,
        "indexes": sorted(city_set),
        "distance_from_my_general": summary_numbers(city_distances_from_me),
        "distance_from_enemy_general": summary_numbers(city_distances_from_enemy),
        "owned_city_count": sum(1 for city in city_set if terrain[city] >= 0),
    }


def spawn_metrics(state, terrain):
    if state.my_general_index is None:
        return None

    transforms = spawn_transform_indexes(state)
    result = {
        "expected_indexes": transforms,
        "actual_enemy_general_index": state.enemy_general_index,
    }

    if state.enemy_general_index is not None:
        distances = build_distance_map(state, state.enemy_general_index, terrain, avoid_cities=False)
        result["distance_between_generals"] = {
            "manhattan": distance_to_target(state, state.my_general_index, state.enemy_general_index),
            "path": distances.get(state.my_general_index),
        }
        result["transform_distances"] = {
            name: distance_to_target(state, index, state.enemy_general_index)
            for name, index in transforms.items()
        }
        result["best_transform"] = min(
            result["transform_distances"],
            key=lambda name: result["transform_distances"][name],
        )

    return result


def spawn_transform_indexes(state):
    my_x = state.my_general_index % state.width
    my_y = state.my_general_index // state.width
    return {
        "rotated_180": xy_to_index(state, state.width - 1 - my_x, state.height - 1 - my_y),
        "horizontal_mirror": xy_to_index(state, state.width - 1 - my_x, my_y),
        "vertical_mirror": xy_to_index(state, my_x, state.height - 1 - my_y),
    }


def symmetry_metrics(state, terrain, city_set):
    return {
        "rotated_180": symmetry_for_transform(
            state,
            terrain,
            city_set,
            lambda x, y: (state.width - 1 - x, state.height - 1 - y),
        ),
        "horizontal_mirror": symmetry_for_transform(
            state,
            terrain,
            city_set,
            lambda x, y: (state.width - 1 - x, y),
        ),
        "vertical_mirror": symmetry_for_transform(
            state,
            terrain,
            city_set,
            lambda x, y: (x, state.height - 1 - y),
        ),
    }


def symmetry_for_transform(state, terrain, city_set, transform):
    compared = 0
    blocker_matches = 0
    city_matches = 0

    for index, tile in enumerate(terrain):
        x = index % state.width
        y = index // state.width
        other_x, other_y = transform(x, y)
        other = xy_to_index(state, other_x, other_y)
        if other < index:
            continue

        compared += 1
        if is_blocker(tile) == is_blocker(terrain[other]):
            blocker_matches += 1
        if (index in city_set) == (other in city_set):
            city_matches += 1

    return {
        "compared_pairs": compared,
        "blocker_similarity": round(blocker_matches / compared, 4) if compared else 0,
        "city_similarity": round(city_matches / compared, 4) if compared else 0,
    }


def is_blocker(tile):
    return tile in (TILE_MOUNTAIN, TILE_FOG_OBSTACLE)


def region_metrics(state, terrain):
    visited = set()
    components = []

    for index, tile in enumerate(terrain):
        if index in visited or not is_passable(tile):
            continue

        queue = deque([index])
        visited.add(index)
        size = 0
        while queue:
            current = queue.popleft()
            size += 1
            for neighbor in state.neighbor_indexes(current):
                if neighbor in visited or not is_passable(terrain[neighbor]):
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        components.append(size)

    components.sort(reverse=True)
    passable_count = sum(components)
    return {
        "passable_component_count": len(components),
        "largest_passable_component": components[0] if components else 0,
        "largest_passable_component_ratio": (
            round(components[0] / passable_count, 4)
            if components and passable_count
            else 0
        ),
        "component_sizes_top5": components[:5],
    }


def mountain_metrics(state, terrain):
    mountain_indexes = [index for index, tile in enumerate(terrain) if tile == TILE_MOUNTAIN]
    edge_mountains = 0
    dead_ends = 0

    for index, tile in enumerate(terrain):
        x = index % state.width
        y = index // state.width
        if tile == TILE_MOUNTAIN and (x in (0, state.width - 1) or y in (0, state.height - 1)):
            edge_mountains += 1
        if is_passable(tile):
            passable_neighbors = sum(1 for neighbor in state.neighbor_indexes(index) if is_passable(terrain[neighbor]))
            if passable_neighbors == 1:
                dead_ends += 1

    return {
        "mountain_count": len(mountain_indexes),
        "edge_mountain_count": edge_mountains,
        "dead_end_passable_tiles": dead_ends,
    }


def quadrant_metrics(state, terrain, city_set):
    quadrants = {
        "top_left": [],
        "top_right": [],
        "bottom_left": [],
        "bottom_right": [],
    }

    mid_x = state.width / 2
    mid_y = state.height / 2
    for index, tile in enumerate(terrain):
        x = index % state.width
        y = index // state.width
        horizontal = "left" if x < mid_x else "right"
        vertical = "top" if y < mid_y else "bottom"
        quadrants[f"{vertical}_{horizontal}"].append((index, tile))

    result = {}
    for name, tiles in quadrants.items():
        count = len(tiles)
        mountains = sum(1 for _, tile in tiles if tile == TILE_MOUNTAIN)
        cities = sum(1 for index, _ in tiles if index in city_set)
        result[name] = {
            "tile_count": count,
            "mountain_count": mountains,
            "mountain_density": round(mountains / count, 4) if count else 0,
            "city_count": cities,
        }

    return result


def army_metrics(armies, terrain):
    visible_armies = [army for army, tile in zip(armies, terrain) if tile >= 0 and army > 0]
    return {
        "owned_tiles_with_army": len(visible_armies),
        "army_total_visible_owned": sum(visible_armies),
        "army_max_visible_owned": max(visible_armies) if visible_armies else 0,
    }


def summary_numbers(values):
    if not values:
        return None

    values = sorted(values)
    return {
        "min": values[0],
        "max": values[-1],
        "avg": round(sum(values) / len(values), 2),
    }
