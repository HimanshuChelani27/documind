"""Hands freed memory back to the OS after heavy work (glibc only, i.e. Linux hosts).

Model inference allocates and frees large buffers; glibc keeps them mapped by default, so a
small instance's resident memory creeps toward its limit even though nothing is leaking."""

import ctypes
import sys
from functools import lru_cache


@lru_cache
def _libc():
    if not sys.platform.startswith("linux"):
        return None
    try:
        return ctypes.CDLL("libc.so.6")
    except OSError:
        return None


def trim() -> None:
    libc = _libc()
    if libc is not None:
        libc.malloc_trim(0)
