"""Exact, bounded caches; no rounding of scientific parameters."""
from collections import OrderedDict
import numpy as np


def size_bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (list, tuple)):
        return 128+sum(size_bytes(item) for item in value)
    if isinstance(value, dict):
        return 128+sum(size_bytes(k)+size_bytes(v) for k, v in value.items())
    return getattr(value, 'nbytes', 128)


class ExactLRU:
    def __init__(self, max_bytes=256*1024**2, max_entries=None):
        if max_bytes < 0:
            raise ValueError('Cache budget must be nonnegative')
        self.max_bytes = int(max_bytes)
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.nbytes = self.hits = self.misses = self.evictions = 0

    def get(self, key, default=None):
        if key not in self.entries:
            self.misses += 1
            return default
        self.hits += 1
        value, _ = self.entries[key]
        self.entries.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self.entries:
            self.nbytes -= self.entries.pop(key)[1]
        count = size_bytes(value)
        if count > self.max_bytes:
            return
        while self.entries and (self.nbytes+count > self.max_bytes or
                                (self.max_entries is not None and len(self.entries) >= self.max_entries)):
            _, (_, old_size) = self.entries.popitem(last=False)
            self.nbytes -= old_size
            self.evictions += 1
        self.entries[key] = (value, count)
        self.nbytes += count

    def clear(self):
        self.entries.clear()
        self.nbytes = 0

    def __len__(self):
        return len(self.entries)

    def empty_like(self):
        return type(self)(self.max_bytes, self.max_entries)

    def __getitem__(self, key):
        if key not in self.entries:
            raise KeyError(key)
        return self.get(key)

    def __contains__(self, key):
        return key in self.entries
