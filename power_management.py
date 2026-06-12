import ctypes
import platform


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class WindowsSleepGuard:
    def __init__(self, reason="background work"):
        self.reason = reason
        self.active = False
        self.available = platform.system() == "Windows"

    def set_active(self, active):
        if active == self.active:
            return

        if active:
            self.prevent_sleep()
        else:
            self.allow_sleep()

    def prevent_sleep(self):
        if not self.available:
            self.active = True
            return

        result = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        if result:
            self.active = True

    def allow_sleep(self):
        if not self.available:
            self.active = False
            return

        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        self.active = False

    def __enter__(self):
        self.prevent_sleep()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.allow_sleep()
