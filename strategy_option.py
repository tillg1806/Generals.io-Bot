from dataclasses import dataclass


@dataclass
class StrategyOption:
    name: str = "expand"
    reason: str | None = None
    started_at_visible_turn: int | None = None
