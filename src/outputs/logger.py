from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_file: Path, to_console: bool = True) -> None:
    handlers = [logging.FileHandler(log_file, encoding="utf-8")]
    if to_console:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
