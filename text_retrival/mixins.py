import time
from typing import Callable

# ---------- Mixins ----------
class LogMixin:
    def log(self, *args):  # lightweight
        print("[LOG]", *args)

class TimeMixin:
    def timeit(self, label: str, fn: Callable, *a, **k):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        dt = (time.perf_counter() - t0) * 1000
        print(f"[TIMER] {label}: {dt:.2f} ms")
        return out