"""Model-independent exact-frame preprocessing and optional display-only sensor delivery.

No CARLA, Torch, camera names or image operations here. Callers supply transforms
and an optional combine function. They own the executor and must await its results.
Required model sensors always use the requested frame; optional display sensors may lag.
"""
from collections import deque
from queue import Queue
from threading import Lock
import time


class TimedSensorQueue(Queue):
    def __init__(self, pool, transforms=None, combine=None, optional_tags=()):
        super().__init__()
        self.pool = pool
        self.transforms = transforms or {}
        self.combine = combine
        self.optional_tags = frozenset(optional_tags)
        if self.optional_tags.intersection(self.transforms):
            raise ValueError('Display-only sensors cannot be preprocessing dependencies')
        self.pipeline = bool(self.transforms)
        self.arrivals = {}
        self._optional = {tag: deque(maxlen=4) for tag in self.optional_tags}
        self._optional_lock = Lock()
        self.begin(0)

    def begin(self, frame):
        self.target = frame
        self.started = time.perf_counter()
        self.futures = {}
        self.combined_future = None
        self.trace = {}

    def put(self, item, block=True, timeout=None):
        tag, frame, data = item
        if tag in self.optional_tags:
            with self._optional_lock:
                self._optional[tag].append((frame, data))
            return
        self.arrivals[(tag, frame)] = time.perf_counter()
        if len(self.arrivals) > 2048:
            for key in list(self.arrivals)[:1024]:
                self.arrivals.pop(key, None)
        return super().put(item, block, timeout)

    def optional_frame(self, tag, frame):
        """Latest available display frame <= requested frame, or None. Never blocks."""
        with self._optional_lock:
            matches = [item for item in self._optional[tag] if item[0] <= frame]
            return max(matches, key=lambda item: item[0]) if matches else None

    def get(self, block=True, timeout=None):
        item = super().get(block, timeout)
        tag, frame, data = item
        now = time.perf_counter()
        arrived = self.arrivals.pop((tag, frame), now)
        if frame == self.target:
            self.trace[tag] = dict(arrival_ms=(arrived - self.started) * 1000,
                                   queue_ms=(now - arrived) * 1000)
            if tag in self.transforms:
                self.futures[tag] = self.pool.submit(self.transforms[tag], data)
                if self.combine and self.transforms.keys() <= self.futures.keys():
                    # All dependencies are queued before the combining task.
                    self.combined_future = self.pool.submit(self.combine, dict(self.futures))
        return item


def collect_frame(queue, required_tags, frame, timeout=300):
    """Exact-frame barrier for explicit required sensors, excluding optional display feeds.

Mirrors the official collector's frame filtering and per-receive timeout. Empty/timeout
propagates to the host adapter, which translates it to its simulator-specific exception.
"""
    required = frozenset(required_tags)
    if required.intersection(queue.optional_tags):
        raise ValueError('An optional sensor cannot also be required')
    result = {}
    while result.keys() < required:
        tag, arrived_frame, data = queue.get(True, timeout)
        if arrived_frame == frame and tag in required:
            result[tag] = (arrived_frame, data)
    return result
