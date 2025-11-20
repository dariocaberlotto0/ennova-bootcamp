from typing import Callable, Dict, List, Any

# ---------- Observer ----------
class EventBus:
    def __init__(self):
        self._subs: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
    def subscribe(self, event: str, fn: Callable[[Dict[str, Any]], None]):
        self._subs.setdefault(event, []).append(fn)
    def publish(self, event: str, payload: Dict[str, Any]):
        for fn in self._subs.get(event, []):
            fn(payload)