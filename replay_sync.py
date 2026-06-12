import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_REPLAY_INDEXES = (
    Path("data/replays/bot_replays.json"),
    Path("data/replays/pybot_replays.json"),
)
DEFAULT_RESULT_GLOB = "runs/**/*_result.json"
DEFAULT_RAW_DIR = Path("data/replays/raw")
DEFAULT_DECODED_DIR = Path("data/replays/decoded")
MAX_REPLAY_BYTES = 1_000_000_000
REPLAY_URL_PATTERNS = (
    "https://generalsio-replays-bot.s3.amazonaws.com/{replay_id}.gior",
    "https://generalsio-replays-bot.s3.amazonaws.com/{replay_id}",
    "https://s3.amazonaws.com/generalsio-replays-bot/{replay_id}.gior",
    "https://s3.amazonaws.com/generalsio-replays-bot/{replay_id}",
)


def sync_replays(
    max_bytes=MAX_REPLAY_BYTES,
    raw_dir=DEFAULT_RAW_DIR,
    decoded_dir=DEFAULT_DECODED_DIR,
    max_download_attempts=25,
):
    raw_dir = Path(raw_dir)
    decoded_dir = Path(decoded_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    decoded_dir.mkdir(parents=True, exist_ok=True)

    replay_ids = collect_replay_ids()
    existing_bytes = directory_size(raw_dir) + directory_size(decoded_dir)
    remaining_bytes = max(0, int(max_bytes) - existing_bytes)
    result = {
        "status": "ok",
        "known_replay_ids": len(replay_ids),
        "downloaded": 0,
        "converted": 0,
        "skipped_existing": 0,
        "failed": 0,
        "attempted": 0,
        "bytes_used": existing_bytes,
        "bytes_remaining": remaining_bytes,
    }

    for replay_id in replay_ids:
        raw_path = raw_dir / f"{replay_id}.gior"
        decoded_path = decoded_dir / f"{replay_id}.gioreplay"
        if raw_path.exists():
            result["skipped_existing"] += 1
        elif remaining_bytes <= 0:
            result["status"] = "storage_limit_reached"
            break
        else:
            if result["attempted"] >= max_download_attempts:
                result["status"] = "attempt_budget_used"
                break
            result["attempted"] += 1
            download = download_replay(replay_id, raw_path, remaining_bytes)
            if download["status"] == "downloaded":
                result["downloaded"] += 1
                remaining_bytes -= download["bytes"]
            else:
                result["failed"] += 1
                continue

        if raw_path.exists() and not decoded_path.exists():
            if convert_gior(raw_path, decoded_path):
                result["converted"] += 1

    result["bytes_used"] = directory_size(raw_dir) + directory_size(decoded_dir)
    result["bytes_remaining"] = max(0, int(max_bytes) - result["bytes_used"])
    return result


def collect_replay_ids():
    replay_ids = set()
    for path in DEFAULT_REPLAY_INDEXES:
        replay_ids.update(replay_ids_from_json(path))

    for path in Path(".").glob(DEFAULT_RESULT_GLOB):
        replay_ids.update(replay_ids_from_json(path))

    return sorted(replay_ids)


def replay_ids_from_json(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

    replay_ids = set()

    def visit(value):
        if isinstance(value, dict):
            replay_id = value.get("replay_id")
            if replay_id:
                replay_ids.add(str(replay_id))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(data)
    return replay_ids


def download_replay(replay_id, target_path, byte_limit):
    for pattern in REPLAY_URL_PATTERNS:
        url = pattern.format(replay_id=replay_id)
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                content_type = response.headers.get("Content-Type", "")
                payload = response.read(byte_limit + 1)
        except (OSError, urllib.error.URLError, urllib.error.HTTPError):
            continue

        if len(payload) > byte_limit:
            return {"status": "too_large", "bytes": len(payload), "url": url}
        if looks_like_html(payload, content_type):
            continue
        if not payload:
            continue

        Path(target_path).write_bytes(payload)
        return {"status": "downloaded", "bytes": len(payload), "url": url}

    return {"status": "not_found", "bytes": 0, "url": None}


def convert_gior(raw_path, decoded_path):
    try:
        import lzstring
    except ImportError:
        return False

    raw_text = Path(raw_path).read_text(encoding="utf-8", errors="ignore")
    decoded = None
    lz = lzstring.LZString()
    for decoder in (
        lz.decompressFromEncodedURIComponent,
        lz.decompressFromBase64,
        lz.decompress,
    ):
        try:
            decoded = decoder(raw_text)
        except Exception:
            decoded = None
        if decoded:
            break

    if not decoded:
        return False

    target = Path(decoded_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(decoded, encoding="utf-8")
    return True


def directory_size(path):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def looks_like_html(payload, content_type):
    if "html" in (content_type or "").lower():
        return True
    prefix = payload[:128].lstrip().lower()
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")


def clear_old_start_scripts():
    legacy_dir = Path("scripts/legacy")
    legacy_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for path in Path("scripts").glob("*"):
        if path.is_file() and path.suffix.lower() in {".ps1", ".bat", ".cmd", ".sh"}:
            target = legacy_dir / path.name
            shutil.move(str(path), str(target))
            moved.append(str(target))
    return moved
