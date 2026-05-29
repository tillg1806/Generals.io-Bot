import json
from pathlib import Path

from config import TRAINING_ACCOUNTS_FILE


class TrainingAccountPool:
    def __init__(self, path=TRAINING_ACCOUNTS_FILE):
        self.path = Path(path)
        self.data = self._load()

    def _load(self):
        if not self.path.exists():
            return {"next": 1, "accounts": []}

        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"next": 1, "accounts": []}

    def save(self):
        self.path.write_text(
            json.dumps(self.data, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def ensure_accounts(self, count, user_id_prefix, username_prefix):
        accounts = [
            account
            for account in self.data.setdefault("accounts", [])
            if account.get("username", "").startswith(username_prefix)
            and not account.get("blocked", False)
        ]
        next_number = int(self.data.get("next", 1))

        while len(accounts) < count:
            accounts.append({
                "user_id": f"{user_id_prefix}_account_{next_number:06d}",
                "username": f"{username_prefix}{next_number:06d}",
            })
            next_number += 1

        self.data["accounts"] = accounts
        self.data["next"] = next_number
        self.save()
        return accounts[:count]

    def mark_blocked(self, usernames, reason):
        blocked_usernames = set(usernames)
        changed = False

        for account in self.data.setdefault("accounts", []):
            if account.get("username") in blocked_usernames:
                account["blocked"] = True
                account["blocked_reason"] = reason
                changed = True

        if changed:
            self.save()

    def mark_usable(self, usernames):
        usable_usernames = set(usernames)
        changed = False

        for account in self.data.setdefault("accounts", []):
            if account.get("username") in usable_usernames:
                account["usable"] = True
                account.pop("blocked", None)
                account.pop("blocked_reason", None)
                changed = True

        if changed:
            self.save()
