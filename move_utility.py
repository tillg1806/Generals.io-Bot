from dataclasses import dataclass


@dataclass
class MoveUtility:
    score: int
    components: dict[str, int]
