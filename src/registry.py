from __future__ import annotations

from typing import Callable, Dict


class Registry:
    def __init__(self) -> None:
        self._items: Dict[str, Callable] = {}

    def register(self, name: str, item: Callable) -> None:
        if name in self._items:
            raise ValueError(f"Item '{name}' is already registered.")
        self._items[name] = item

    def get(self, name: str) -> Callable:
        if name not in self._items:
            raise KeyError(f"Unknown registry item: {name}")
        return self._items[name]

    def keys(self):
        return self._items.keys()
