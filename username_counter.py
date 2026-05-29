import json
from pathlib import Path

from config import USERNAME_COUNTER_FILE


class UsernameCounter:
    def __init__(self, path=USERNAME_COUNTER_FILE):
        self.path = Path(path)
        self.data = self._load()

    def _load(self):
        if not self.path.exists():
            return {"next": 1}

        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"next": 1}

    def save(self):
        self.path.write_text(
            json.dumps(self.data, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def reserve(self, count, prefix):
        start = int(self.data.get("next", 1))
        names = [f"{prefix}_{number:06d}" for number in range(start, start + count)]
        self.data["next"] = start + count
        self.save()
        return names
