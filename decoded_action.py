from dataclasses import dataclass


@dataclass(frozen=True)
class DecodedAction:
    source: int | None
    target: int | None
    half: bool
    is_pass: bool
    direction: int | None
