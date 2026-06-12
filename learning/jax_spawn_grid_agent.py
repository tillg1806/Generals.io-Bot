from pathlib import Path

from learning.jax_agent import SPAWN_GRID_MODEL_FILE, read_json


def same_conv2d(inputs, weights, dilation=1):
    import numpy as np

    batch, height, width, in_channels = inputs.shape
    kernel_height, kernel_width, _, out_channels = weights.shape
    pad_y = dilation * (kernel_height - 1) // 2
    pad_x = dilation * (kernel_width - 1) // 2
    padded = np.pad(
        inputs,
        ((0, 0), (pad_y, pad_y), (pad_x, pad_x), (0, 0)),
        mode="constant",
    )
    output = np.zeros((batch, height, width, out_channels), dtype=np.float32)
    for ky in range(kernel_height):
        for kx in range(kernel_width):
            source = padded[
                :,
                ky * dilation:ky * dilation + height,
                kx * dilation:kx * dilation + width,
                :,
            ]
            output += np.tensordot(source, weights[ky, kx], axes=([3], [0]))
    return output


class JaxSpawnGridAgent:
    def __init__(self, path=SPAWN_GRID_MODEL_FILE, score_cap=20):
        self.path = Path(path)
        self.score_cap = score_cap
        self.force_enabled = False
        self.model = read_json(self.path, None)
        if not isinstance(self.model, dict):
            self.model = None

    def is_trained(self):
        return bool(self.model and self.model.get("params") and self.model.get("channels"))

    def is_ready(self):
        status = self.model.get("status") if isinstance(self.model, dict) else {}
        return self.is_trained() and (
            self.force_enabled
            or status.get("use_in_general_guesser") is True
        )

    def score_map(self, state, terrain, score_cap=None):
        import numpy as np

        if not self.is_ready():
            return {}
        if state.my_general_index is None or state.width <= 0 or state.height <= 0:
            return {}
        if len(terrain or []) < state.width * state.height:
            return {}

        grid, mask = self.state_to_grid(state, terrain)
        logits = self.forward_numpy(grid.reshape(1, *grid.shape)).reshape(-1)
        logits = logits.copy()
        logits[~mask] = -1.0e9
        valid_logits = logits[mask]
        if valid_logits.size == 0:
            return {}

        shifted = logits - float(valid_logits.max())
        exp = np.exp(shifted)
        exp[~mask] = 0.0
        total = float(exp.sum()) or 1.0
        probabilities = exp / total
        max_probability = float(probabilities[mask].max()) or 1.0
        cap = self.score_cap if score_cap is None else score_cap
        grid_size = int(self.model.get("grid_size") or 24)
        scores = {}
        for row in range(min(state.height, grid_size)):
            for col in range(min(state.width, grid_size)):
                grid_index = row * grid_size + col
                probability = probabilities[grid_index]
                if probability <= 0:
                    continue
                game_index = row * state.width + col
                scores[game_index] = {
                    "spawn_grid_probability": round(float(probability), 6),
                    "score_adjustment": int((float(probability) / max_probability) * cap),
                }
        return scores

    def state_to_grid(self, state, terrain):
        import numpy as np

        grid_size = int(self.model.get("grid_size") or 24)
        channels = self.model.get("channels") or []
        grid = np.zeros((grid_size, grid_size, len(channels)), dtype=np.float32)
        mask = np.zeros((grid_size * grid_size,), dtype=np.bool_)
        width = state.width
        height = state.height
        my_general = state.my_general_index
        city_set = state.city_set()
        my_x = my_general % width
        my_y = my_general // width
        rotated = (height - 1 - my_y) * width + (width - 1 - my_x)
        horizontal = my_y * width + (width - 1 - my_x)
        vertical = (height - 1 - my_y) * width + my_x
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        max_edge = max(1.0, min(width, height) / 2.0)
        scale = max(1.0, float(width + height))

        for row in range(min(height, grid_size)):
            for col in range(min(width, grid_size)):
                index = row * width + col
                tile = terrain[index]
                is_mountain = tile in (-2, -4)
                is_city = index in city_set
                is_my_general = index == my_general
                is_valid_candidate = (
                    not is_mountain
                    and not is_city
                    and not is_my_general
                    and tile != state.player_index
                    and not state.has_seen(index)
                )
                grid_index = row * grid_size + col
                mask[grid_index] = is_valid_candidate
                edge_distance = min(col, width - 1 - col, row, height - 1 - row) / max_edge
                center_distance = (abs(col - center_x) + abs(row - center_y)) / scale
                values = {
                    "mountain": 1.0 if is_mountain else 0.0,
                    "city": 1.0 if is_city else 0.0,
                    "my_general": 1.0 if is_my_general else 0.0,
                    "valid_candidate": 1.0 if is_valid_candidate else 0.0,
                    "rotated_hint": self.inverse_hint(width, height, index, rotated),
                    "horizontal_hint": self.inverse_hint(width, height, index, horizontal),
                    "vertical_hint": self.inverse_hint(width, height, index, vertical),
                    "edge_distance": edge_distance,
                    "center_distance": center_distance,
                    "x_position": col / max(1.0, width - 1),
                    "y_position": row / max(1.0, height - 1),
                    "width": width / grid_size,
                    "height": height / grid_size,
                }
                for channel_index, channel in enumerate(channels):
                    grid[row, col, channel_index] = values.get(channel, 0.0)
        return grid, mask

    def inverse_hint(self, width, height, index, target):
        scale = max(1.0, float(width + height))
        x = index % width
        y = index // width
        target_x = target % width
        target_y = target // width
        distance = abs(x - target_x) + abs(y - target_y)
        return max(0.0, 1.0 - distance / scale)

    def forward_numpy(self, grid):
        import numpy as np

        params = self.model.get("params") or {}
        architecture = self.model.get("architecture") or {}
        dilations = architecture.get("dilations") or [1]
        residual = architecture.get("residual", True)
        x = grid.astype(np.float32)
        for layer_index, layer in enumerate(params.get("layers") or []):
            previous = x
            weights = np.array(layer["w"], dtype=np.float32)
            bias = np.array(layer["b"], dtype=np.float32)
            dilation = int(dilations[layer_index % len(dilations)])
            x = same_conv2d(x, weights, dilation=dilation) + bias
            if residual and layer_index > 0 and previous.shape == x.shape:
                x = x + previous
            x = np.maximum(x, 0.0)
        out_w = np.array(params["out_w"], dtype=np.float32)
        out_b = np.array(params["out_b"], dtype=np.float32)
        return same_conv2d(x, out_w, dilation=1) + out_b
