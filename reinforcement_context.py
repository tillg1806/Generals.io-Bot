from dataclasses import dataclass


@dataclass
class ReinforcementContext:
    turns_until: int
    turns_since: int
    staging: bool
    attack_window: bool
    expand_window: bool
    waypoint: int | None = None
