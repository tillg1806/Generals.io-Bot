from dataclasses import dataclass


@dataclass
class Move:
    source: int
    target: int
    half: bool
    strategy_target: int | None
