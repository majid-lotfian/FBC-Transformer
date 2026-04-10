from __future__ import annotations


def chunked(items, size: int):
    for idx in range(0, len(items), size):
        yield items[idx:idx + size]
