from dataclasses import dataclass


@dataclass
class PrioritizedMove:
    score: int
    reason: str
