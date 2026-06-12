import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from replay_sync import sync_replays


REPO_DIR = Path(__file__).resolve().parent
SYNC_INTERVAL_SECONDS = 90


class Menu:
    def __init__(self, title, options):
        self.title = title
        self.options = options
        self.index = 0

    def choose(self):
        while True:
            self.render()
            key = read_key()
            if key in ("up", "left"):
                self.index = (self.index - 1) % len(self.options)
            elif key in ("down", "right"):
                self.index = (self.index + 1) % len(self.options)
            elif key == "enter":
                return self.options[self.index]
            elif key in ("q", "esc"):
                return None

    def render(self):
        clear_screen()
        print(self.title)
        print("")
        print("Pfeiltasten: auswaehlen | Enter: starten | q: abbrechen")
        print("")
        for index, option in enumerate(self.options):
            prefix = ">" if index == self.index else " "
            print(f"{prefix} {option['label']}")


class LauncherActionMenu:
    def __init__(self):
        self.index = 0
        self.requeue = True
        self.actions = [
            {
                "label": "Public 1v1 Queue",
                "command": ["main.py", "--queue", "1v1"],
                "sync": True,
                "live_bot": True,
            },
            {
                "label": "Private 1v1 Queue / Custom Room",
                "command": ["main.py", "--queue", "private"],
                "sync": True,
                "live_bot": True,
            },
            {
                "label": "Simulator Test",
                "command": [
                    "main.py",
                    "--sim-self-play",
                    "--sim-games",
                    "32",
                    "--sim-parallel-games",
                    "16",
                    "--sim-grid-size",
                    "10",
                    "--sim-truncation",
                    "500",
                    "--sim-opponent",
                    "expander",
                ],
                "sync": False,
            },
            {
                "label": "Replay Sync jetzt ausfuehren",
                "command": None,
                "sync_only": True,
            },
        ]

    def choose(self):
        while True:
            self.render()
            key = read_key()
            if key in ("up", "left"):
                self.index = (self.index - 1) % self.option_count()
            elif key in ("down", "right"):
                self.index = (self.index + 1) % self.option_count()
            elif key == "enter":
                if self.index == 0:
                    self.requeue = not self.requeue
                    continue
                action = dict(self.actions[self.index - 1])
                action["requeue"] = self.requeue
                if action.get("live_bot") and not self.requeue:
                    action["command"] = [*action["command"], "--no-requeue"]
                return action
            elif key in ("q", "esc"):
                return None

    def option_count(self):
        return len(self.actions) + 1

    def render(self):
        clear_screen()
        print("TillBot Launcher - Aktion")
        print("")
        print("Pfeiltasten: auswaehlen | Enter: toggeln/starten | q: abbrechen")
        print("")
        labels = [f"Requeue: {'AN' if self.requeue else 'AUS'}"]
        labels.extend(action["label"] for action in self.actions)
        for index, label in enumerate(labels):
            prefix = ">" if index == self.index else " "
            print(f"{prefix} {label}")


def main():
    environment = Menu(
        "TillBot Launcher - Umgebung",
        [
            {"label": "Lokal / Windows Python", "value": "local"},
            {"label": "WSL", "value": "wsl"},
        ],
    ).choose()
    if environment is None:
        return

    action = LauncherActionMenu().choose()
    if action is None:
        return

    clear_screen()
    if action.get("sync_only"):
        print_sync_result(run_replay_sync(max_attempts=25))
        return

    stop_sync = threading.Event()
    sync_thread = None
    sync_status = {
        "last_result": None,
        "last_run_at": None,
    }
    if action.get("sync"):
        sync_status["last_result"] = run_replay_sync(max_attempts=5)
        sync_status["last_run_at"] = datetime.now().strftime("%H:%M:%S")
        sync_thread = threading.Thread(
            target=sync_loop,
            args=(stop_sync, sync_status),
            daemon=True,
            name="replay-sync",
        )
        sync_thread.start()

    command = build_command(environment["value"], action["command"])
    log_path = launcher_log_path(action, environment["value"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    start_log_console(log_path)

    process = None
    log_handle = None
    try:
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        write_log_header(log_handle, command, action, environment)
        process = subprocess.Popen(
            command,
            cwd=REPO_DIR,
            stdin=subprocess.PIPE,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        run_dashboard(process, command, log_path, action, environment, sync_status)
    finally:
        stop_sync.set()
        if sync_thread:
            sync_thread.join(timeout=3)
        if log_handle:
            log_handle.close()
        if action.get("sync"):
            final_sync = run_replay_sync(max_attempts=10)
            sync_status["last_result"] = final_sync
            sync_status["last_run_at"] = datetime.now().strftime("%H:%M:%S")
            clear_screen()
            print("Bot beendet.")
            print("")
            print_sync_result(final_sync, prefix="Replay-Sync nach Ende: ")
            print("")
            print(f"Log: {log_path}")


def build_command(environment, args):
    if environment == "local":
        return [sys.executable, *args]

    wsl_cwd = windows_to_wsl_path(REPO_DIR)
    shell_command = " && ".join(
        [
            "cd " + quote_sh(wsl_cwd),
            "test -f .venv_wsl/bin/activate || python3 -m venv .venv_wsl",
            ". .venv_wsl/bin/activate",
            "python -m pip install -q --upgrade pip",
            "python -m pip install -q -r requirements_pybot.txt",
            "python " + " ".join(quote_sh(arg) for arg in args),
        ]
    )
    return ["wsl", "bash", "-lc", shell_command]


def windows_to_wsl_path(path):
    path = Path(path).resolve()
    drive = path.drive.rstrip(":").lower()
    rest = path.as_posix().split(":", 1)[-1].lstrip("/")
    return f"/mnt/{drive}/{rest}"


def quote_sh(value):
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def sync_loop(stop_event, sync_status):
    while not stop_event.wait(SYNC_INTERVAL_SECONDS):
        result = run_replay_sync(max_attempts=5)
        sync_status["last_result"] = result
        sync_status["last_run_at"] = datetime.now().strftime("%H:%M:%S")


def run_replay_sync(max_attempts=5):
    try:
        return sync_replays(max_download_attempts=max_attempts)
    except Exception as error:
        return {"status": "error", "error": str(error)}


def print_sync_result(result, prefix=""):
    if result.get("status") == "error":
        print(f"{prefix}Fehler: {result.get('error')}")
        return
    print(
        f"{prefix}Status: {result.get('status')} | "
        f"IDs: {result.get('known_replay_ids')} | "
        f"neu: {result.get('downloaded')} | "
        f"konvertiert: {result.get('converted')} | "
        f"fehlgeschlagen: {result.get('failed')} | "
        f"frei: {format_bytes(result.get('bytes_remaining', 0))}"
    )


def launcher_log_path(action, environment):
    label = action.get("label", "run").lower()
    safe_label = "".join(char if char.isalnum() else "_" for char in label).strip("_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_DIR / "logs" / "launcher" / f"{timestamp}_{environment}_{safe_label}.log"


def write_log_header(handle, command, action, environment):
    handle.write("=" * 80 + "\n")
    handle.write(f"TillBot launcher log: {datetime.now().isoformat(timespec='seconds')}\n")
    handle.write(f"Environment: {environment['label']}\n")
    handle.write(f"Action: {action.get('label')}\n")
    handle.write(f"Requeue: {'on' if action.get('requeue', True) else 'off'}\n")
    handle.write("Command: " + " ".join(command) + "\n")
    handle.write("=" * 80 + "\n\n")


def start_log_console(log_path):
    if os.name != "nt":
        return None

    command = [
        "powershell",
        "-NoExit",
        "-Command",
        (
            "$Host.UI.RawUI.WindowTitle = 'TillBot Log'; "
            f"Write-Host 'TillBot log: {escape_powershell(str(log_path))}' -ForegroundColor Cyan; "
            f"Get-Content -LiteralPath '{escape_powershell(str(log_path))}' -Wait -Tail 80"
        ),
    ]
    try:
        return subprocess.Popen(
            command,
            cwd=REPO_DIR,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    except OSError:
        return None


def escape_powershell(value):
    return str(value).replace("'", "''")


def run_dashboard(process, command, log_path, action, environment, sync_status):
    started_at = time.time()
    status_message = "laeuft"
    while True:
        return_code = process.poll()
        if return_code is not None:
            status_message = f"beendet (Exit {return_code})"
        render_dashboard(
            status_message,
            started_at,
            command,
            log_path,
            action,
            environment,
            sync_status,
            return_code,
        )

        if return_code is not None:
            wait_for_dashboard_exit()
            return

        key = read_key_nonblocking()
        if key == "enter":
            action_choice = DashboardActionMenu().choose()
            if action_choice == "sync":
                sync_status["last_result"] = run_replay_sync(max_attempts=5)
                sync_status["last_run_at"] = datetime.now().strftime("%H:%M:%S")
            elif action_choice == "stop_after_current":
                send_process_command(process, "e")
                status_message = "stop nach aktuellem Spiel angefordert"
            elif action_choice == "stop_now":
                send_process_command(process, "q")
                terminate_if_still_running(process, grace_seconds=5)
                status_message = "stop angefordert"
            elif action_choice == "back":
                pass

        time.sleep(1)


def render_dashboard(status, started_at, command, log_path, action, environment, sync_status, return_code):
    clear_screen()
    elapsed = int(time.time() - started_at)
    print("TillBot Dashboard")
    print("")
    print(f"Status: {status}")
    print(f"Laufzeit: {elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}")
    print(f"Umgebung: {environment['label']}")
    print(f"Aktion: {action.get('label')}")
    if action.get("live_bot"):
        print(f"Requeue: {'AN' if action.get('requeue', True) else 'AUS'}")
    print("")
    print("Command:")
    print(" ".join(command))
    print("")
    print(f"Log-Konsole: separat geoeffnet")
    print(f"Log-Datei: {log_path}")
    print("")
    print("Replay-Sync:")
    result = sync_status.get("last_result")
    if result:
        print_sync_result(result, prefix=f"  letzter Lauf {sync_status.get('last_run_at')}: ")
    else:
        print("  nicht aktiv")
    print("")
    if return_code is None:
        print("Controls: Enter = Aktionen")
    else:
        print("Enter/q = Dashboard schliessen")


def wait_for_dashboard_exit():
    while True:
        key = read_key()
        if key in ("enter", "q", "esc"):
            return


class DashboardActionMenu:
    def __init__(self):
        self.index = 0
        self.options = [
            {
                "label": "Zurueck zum Dashboard",
                "value": "back",
            },
            {
                "label": "Replay-Sync jetzt",
                "value": "sync",
            },
            {
                "label": "Nach aktuellem Spiel stoppen (e)",
                "value": "stop_after_current",
            },
            {
                "label": "Sofort stoppen (q)",
                "value": "stop_now",
            },
        ]

    def choose(self):
        while True:
            self.render()
            key = read_key()
            if key in ("up", "left"):
                self.index = (self.index - 1) % len(self.options)
            elif key in ("down", "right"):
                self.index = (self.index + 1) % len(self.options)
            elif key == "enter":
                return self.options[self.index]["value"]
            elif key in ("q", "esc"):
                return "back"

    def render(self):
        clear_screen()
        print("TillBot Dashboard - Aktionen")
        print("")
        print("Pfeiltasten: auswaehlen | Enter: ausfuehren | q: zurueck")
        print("")
        for index, option in enumerate(self.options):
            prefix = ">" if index == self.index else " "
            print(f"{prefix} {option['label']}")


def send_process_command(process, command):
    if process.stdin is None:
        return False


def terminate_if_still_running(process, grace_seconds=5):
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.2)
    if process.poll() is None:
        process.terminate()
    try:
        process.stdin.write(command + "\n")
        process.stdin.flush()
        return True
    except OSError:
        return False


def read_key_nonblocking():
    if os.name == "nt":
        import msvcrt

        if not msvcrt.kbhit():
            return None
        return read_key()

    import select

    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None
    return read_key()


def format_bytes(value):
    value = float(value or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def read_key():
    if os.name == "nt":
        import msvcrt

        key = msvcrt.getch()
        if key in (b"\x00", b"\xe0"):
            code = msvcrt.getch()
            return {
                b"H": "up",
                b"P": "down",
                b"K": "left",
                b"M": "right",
            }.get(code, "")
        if key in (b"\r", b"\n"):
            return "enter"
        if key == b"\x1b":
            return "esc"
        try:
            return key.decode("utf-8").lower()
        except UnicodeDecodeError:
            return ""

    import termios
    import tty

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
        if key == "\x1b":
            sequence = sys.stdin.read(2)
            return {
                "[A": "up",
                "[B": "down",
                "[D": "left",
                "[C": "right",
            }.get(sequence, "esc")
        if key in ("\r", "\n"):
            return "enter"
        return key.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


if __name__ == "__main__":
    main()
