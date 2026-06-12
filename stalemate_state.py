from dataclasses import dataclass


@dataclass
class StalemateState:
    active: bool = False
    activated_at_visible_turn: int | None = None
    last_progress_visible_turn: int | None = None
    last_tile_count: int = 0
    last_seen_count: int = 0
    last_target_distance: int | None = None
    repeated_target_count: int = 0
    last_strategy_target: int | None = None
    reason: str | None = None
