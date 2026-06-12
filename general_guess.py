from dataclasses import dataclass


@dataclass
class GeneralGuess:
    index: int
    score: float
    confidence: float
    reason: str
    candidates: list[dict]
