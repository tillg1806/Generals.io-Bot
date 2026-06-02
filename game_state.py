from dataclasses import dataclass, field

from config import (
    ENEMY_MOVEMENT_MEMORY,
    ENEMY_PREDICTION_STEPS,
    INTERNAL_TURNS_PER_VISIBLE_TURN,
    TILE_MOUNTAIN,
)


def apply_diff(old, diff):
    patched = []
    old_index = 0
    diff_index = 0

    while diff_index < len(diff):
        matching = diff[diff_index]
        diff_index += 1

        patched.extend(old[old_index:old_index + matching])
        old_index += matching

        if diff_index >= len(diff):
            break

        mismatch_count = diff[diff_index]
        diff_index += 1
        if mismatch_count == 0:
            continue

        new_values = diff[diff_index:diff_index + mismatch_count]
        patched.extend(new_values)
        old_index += mismatch_count
        diff_index += mismatch_count

    patched.extend(old[old_index:])
    return patched


@dataclass
class GameState:
    player_index: int | None = None
    enemy_player_index: int | None = None
    my_general_index: int | None = None
    enemy_general_index: int | None = None
    map_data: list[int] = field(default_factory=list)
    cities: list[int] = field(default_factory=list)
    visible_enemy_tiles: list[int] = field(default_factory=list)
    scores: list[dict] = field(default_factory=list)
    seen_tiles: set[int] = field(default_factory=set)
    last_seen_turn: dict[int, int] = field(default_factory=dict)
    enemy_movement_heat: dict[int, int] = field(default_factory=dict)
    enemy_movement_last_seen: dict[int, int] = field(default_factory=dict)
    enemy_prediction_heat: dict[int, int] = field(default_factory=dict)
    enemy_prediction_last_seen: dict[int, int] = field(default_factory=dict)
    enemy_attack_events: list[dict] = field(default_factory=list)
    width: int = 0
    height: int = 0
    turn: int = -1
    last_move_turn: int = -1
    expansion_started: bool = False

    def start(self, data):
        self.player_index = data.get("playerIndex", self.player_index)

    def update(self, data):
        self.scores = data.get("scores", self.scores)
        old_terrain, old_armies = self.split_map()

        if "map_diff" in data:
            self.map_data = apply_diff(self.map_data, data["map_diff"])
        elif "map" in data:
            self.map_data = data["map"]

        if "cities_diff" in data:
            self.cities = apply_diff(self.cities, data["cities_diff"])
        elif "cities" in data:
            self.cities = data["cities"]

        if len(self.map_data) >= 2:
            self.width = self.map_data[0]
            self.height = self.map_data[1]

        generals = data.get("generals", [])
        if generals:
            if self.player_index is not None and generals[self.player_index] != -1:
                self.my_general_index = generals[self.player_index]
            elif self.my_general_index is None:
                self.my_general_index = next((g for g in generals if g != -1), None)

            if self.enemy_player_index is None and self.player_index is not None:
                self.enemy_player_index = next(
                    (i for i in range(len(generals)) if i != self.player_index),
                    None,
                )

            if self.enemy_player_index is not None and generals[self.enemy_player_index] != -1:
                self.enemy_general_index = generals[self.enemy_player_index]

        terrain, armies = self.split_map()
        turn = data.get("turn", self.turn)
        self.turn = turn
        self.enemy_attack_events = []
        self.update_seen_tiles(terrain, turn)
        self.update_enemy_movement_heat(old_terrain, old_armies, terrain, armies, turn)

        if self.player_index is not None:
            self.visible_enemy_tiles = [
                index
                for index, owner in enumerate(terrain)
                if owner >= 0 and owner != self.player_index
            ]

    def split_map(self):
        if len(self.map_data) < 2:
            return [], []

        tile_count = self.width * self.height
        armies = self.map_data[2:2 + tile_count]
        terrain = self.map_data[2 + tile_count:2 + tile_count * 2]
        return terrain, armies

    def update_seen_tiles(self, terrain, turn):
        for index, owner in enumerate(terrain):
            if owner in (-3, -4):
                continue

            self.seen_tiles.add(index)
            self.last_seen_turn[index] = turn

    def update_enemy_movement_heat(self, old_terrain, old_armies, terrain, armies, turn):
        if self.player_index is None:
            return
        if len(old_terrain) != len(terrain) or len(old_armies) != len(armies):
            return

        for index, old_owner in enumerate(old_terrain):
            if old_owner < 0 or old_owner == self.player_index:
                continue
            if terrain[index] < 0 or terrain[index] == self.player_index:
                continue
            if old_armies[index] - armies[index] < 2:
                continue

            for neighbor in self.neighbor_indexes(index):
                if terrain[neighbor] < 0 or terrain[neighbor] == self.player_index:
                    continue
                if armies[neighbor] <= old_armies[neighbor]:
                    continue

                self.enemy_movement_heat[index] = self.enemy_movement_heat.get(index, 0) + 3
                self.enemy_movement_heat[neighbor] = self.enemy_movement_heat.get(neighbor, 0) + 1
                self.enemy_movement_last_seen[index] = turn
                self.enemy_movement_last_seen[neighbor] = turn
                self.enemy_attack_events.append(
                    {
                        "turn": turn,
                        "source": index,
                        "target": neighbor,
                        "estimated_army": max(0, old_armies[index] - armies[index]),
                        "source_army_before": old_armies[index],
                        "source_army_after": armies[index],
                        "target_army_before": old_armies[neighbor],
                        "target_army_after": armies[neighbor],
                    }
                )
                self.remember_enemy_prediction(index, neighbor, terrain, turn)

        self.decay_enemy_movement_heat(turn)

    def remember_enemy_prediction(self, source, target, terrain, turn):
        if self.width <= 0 or len(terrain) < self.width * self.height:
            return

        source_x = source % self.width
        source_y = source // self.width
        target_x = target % self.width
        target_y = target // self.width
        direction_x = target_x - source_x
        direction_y = target_y - source_y

        if abs(direction_x) + abs(direction_y) != 1:
            return

        for step in range(1, ENEMY_PREDICTION_STEPS + 1):
            predicted_x = target_x + direction_x * step
            predicted_y = target_y + direction_y * step
            if predicted_x < 0 or predicted_x >= self.width:
                break
            if predicted_y < 0 or predicted_y >= self.height:
                break

            predicted = predicted_y * self.width + predicted_x
            if terrain[predicted] == TILE_MOUNTAIN:
                break

            heat = ENEMY_PREDICTION_STEPS - step + 1
            self.enemy_prediction_heat[predicted] = self.enemy_prediction_heat.get(predicted, 0) + heat
            self.enemy_prediction_last_seen[predicted] = turn

    def decay_enemy_movement_heat(self, turn):
        if turn < 0:
            return

        stale_tiles = [
            index
            for index, last_seen in self.enemy_movement_last_seen.items()
            if turn - last_seen > ENEMY_MOVEMENT_MEMORY
        ]
        for index in stale_tiles:
            self.enemy_movement_heat.pop(index, None)
            self.enemy_movement_last_seen.pop(index, None)

        stale_predictions = [
            index
            for index, last_seen in self.enemy_prediction_last_seen.items()
            if turn - last_seen > ENEMY_MOVEMENT_MEMORY
        ]
        for index in stale_predictions:
            self.enemy_prediction_heat.pop(index, None)
            self.enemy_prediction_last_seen.pop(index, None)

    def neighbor_indexes(self, index):
        if self.width <= 0:
            return []

        result = []
        x = index % self.width

        if index - self.width >= 0:
            result.append(index - self.width)
        if index + self.width < self.width * self.height:
            result.append(index + self.width)
        if x > 0:
            result.append(index - 1)
        if x < self.width - 1:
            result.append(index + 1)

        return result

    def has_seen(self, index):
        return index in self.seen_tiles

    def last_seen(self, index):
        return self.last_seen_turn.get(index, -1)

    def city_set(self):
        return set(self.cities)

    def visible_turn(self, turn):
        if turn < 0:
            return turn

        return (turn + INTERNAL_TURNS_PER_VISIBLE_TURN - 1) // INTERNAL_TURNS_PER_VISIBLE_TURN

    def my_tile_count(self):
        if self.player_index is None:
            return 0

        for score in self.scores:
            if score.get("i") == self.player_index:
                return score.get("tiles", 0)

        return 0

    def biggest_enemy_tile_count(self):
        if self.player_index is None:
            return 0

        enemy_counts = [
            score.get("tiles", 0)
            for score in self.scores
            if score.get("i") != self.player_index and not score.get("dead", False)
        ]
        if not enemy_counts:
            return 0

        return max(enemy_counts)

    def my_total_army(self):
        if self.player_index is None:
            return 0

        for score in self.scores:
            if score.get("i") == self.player_index:
                return score.get("total", 0)

        return 0

    def biggest_enemy_total_army(self):
        if self.player_index is None:
            return 0

        enemy_totals = [
            score.get("total", 0)
            for score in self.scores
            if score.get("i") != self.player_index and not score.get("dead", False)
        ]
        if not enemy_totals:
            return 0

        return max(enemy_totals)
