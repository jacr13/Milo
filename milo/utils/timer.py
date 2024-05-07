import time


class Timer:
    def __init__(self) -> None:
        self.start_time: float | None = None
        self.stop_time: float | None = None
        self.running: bool = False

    def start(self) -> "Timer":
        if self.running:
            raise RuntimeError("Timer is already running.")
        self.start_time = time.time()
        self.running = True
        return self

    def stop(self) -> "Timer":
        if not self.running:
            raise RuntimeError("Timer is not running.")
        self.stop_time = time.time()
        self.running = False
        return self

    def running_time(self) -> float:
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        if self.running:
            return time.time() - self.start_time
        elif self.stop_time is None:
            raise ValueError("Timer has not been stopped.")
        return self.stop_time - self.start_time

    def final_time(self) -> float | None:
        if self.start_time is None or self.running or self.stop_time is None:
            return None  # Timer not started or still running / has not been stopped
        return self.stop_time - self.start_time

    def is_running(self) -> bool:
        return self.running
