"""Small filesystem compatibility helpers for TrustProtKG tests."""

from __future__ import annotations

import os
import time
from pathlib import Path


_ORIGINAL_WRITE_TEXT = Path.write_text
_INSTALLED = False


def _safe_write_text(self: Path, data: str, encoding: str | None = None, errors: str | None = None, newline: str | None = None) -> int:
    try:
        return _ORIGINAL_WRITE_TEXT(self, data, encoding=encoding, errors=errors, newline=newline)
    except OSError as exc:
        if exc.errno != 22:
            raise
        tmp = self.with_name(f".{self.name}.{os.getpid()}.tmp")
        written = _ORIGINAL_WRITE_TEXT(tmp, data, encoding=encoding, errors=errors, newline=newline)
        try:
            for _ in range(5):
                try:
                    os.replace(tmp, self)
                    return written
                except PermissionError:
                    time.sleep(0.05)
            if self.exists():
                return written
            os.replace(tmp, self)
            return written
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass


def install_safe_path_write_text() -> None:
    """Install an Errno 22 fallback for repeated Windows text-file rewrites."""
    global _INSTALLED
    if _INSTALLED:
        return
    Path.write_text = _safe_write_text  # type: ignore[method-assign]
    _INSTALLED = True


__all__ = ["install_safe_path_write_text"]
