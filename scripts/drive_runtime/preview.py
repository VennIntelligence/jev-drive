"""Model-independent latest-frame preview. A slow viewer never blocks the agent."""
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import time

import numpy as np

META = 65536
PIXELS = 1920 * 1080 * 3
SLOT = 16 + META + PIXELS
SIZE = 16 + 2 * SLOT


def shared_path(directory):
    key = hashlib.sha256(str(Path(directory).resolve()).encode()).hexdigest()[:16]
    return '/dev/shm/jev-b2d-%d-%s.rgb' % (os.getuid(), key)


class PreviewWriter:
    def __init__(self, directory):
        fd = os.open(shared_path(directory), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.ftruncate(fd, SIZE)
            self.mem = mmap.mmap(fd, SIZE)
        finally:
            os.close(fd)

    def publish(self, state, images):
        arrays = {k: np.ascontiguousarray(v, dtype=np.uint8) for k, v in images.items()}
        specs, offset = {}, 0
        for name, array in arrays.items():
            specs[name] = dict(shape=list(array.shape), offset=offset, size=array.nbytes)
            offset += array.nbytes
        metadata = json.dumps(dict(state=state, images=specs)).encode()
        if len(metadata) > META or offset > PIXELS:
            raise ValueError('Preview exceeds shared memory capacity')
        seq = struct.unpack_from('<Q', self.mem, 0)[0] + 1
        base = 16 + (seq % 2) * SLOT
        # Mark slot busy before modifying it; reader checks slot and global sequence.
        struct.pack_into('<Q', self.mem, base, 0)
        struct.pack_into('<Q', self.mem, base + 8, len(metadata))
        self.mem[base + 16:base + 16 + len(metadata)] = metadata
        start = base + 16 + META
        for name, array in arrays.items():
            pos = start + specs[name]['offset']
            self.mem[pos:pos + array.nbytes] = memoryview(array).cast('B')
        struct.pack_into('<Q', self.mem, base, seq)
        struct.pack_into('<Q', self.mem, 0, seq)

    def close(self):
        self.mem.close()


class PreviewReader:
    def __init__(self, directory):
        fd = os.open(shared_path(directory), os.O_RDONLY)
        try:
            self.mem = mmap.mmap(fd, SIZE, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)
        self.seq = 0

    def read(self):
        seq = struct.unpack_from('<Q', self.mem, 0)[0]
        if not seq or seq == self.seq:
            return None
        base = 16 + (seq % 2) * SLOT
        if struct.unpack_from('<Q', self.mem, base)[0] != seq:
            return None
        length = struct.unpack_from('<Q', self.mem, base + 8)[0]
        if length > META:
            return None
        raw = self.mem[base + 16:base + 16 + length]
        # Copy the whole bounded pixel region before checking for concurrent overwrite.
        pixels = self.mem[base + 16 + META:base + 16 + META + PIXELS]
        if (struct.unpack_from('<Q', self.mem, base)[0] != seq or
                struct.unpack_from('<Q', self.mem, 0)[0] != seq):
            return None
        packet = json.loads(raw)
        packet['pixels'] = pixels
        self.seq = seq
        return packet

    def close(self):
        self.mem.close()


class LivePreview:
    """Small model-independent publisher: RGB arrays + JSON state, bounded live images.

Image names and state fields are an adapter/viewer contract, not enforced here.
Closing marks the last frame ended. The viewer never owns simulator controls.
"""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.writer = PreviewWriter(directory)
        self.log = (self.directory / 'telemetry.jsonl').open('a', buffering=1)
        self.last_log = 0
        self.state, self.images = {}, {}

    def _write_state(self):
        temp = self.directory / 'status.tmp.json'
        temp.write_text(json.dumps(self.state))
        os.replace(str(temp), str(self.directory / 'status.json'))
        self.log.write(json.dumps(self.state) + '\n')

    def publish(self, state, images):
        self.state, self.images = state, images
        self.writer.publish(state, images)
        if time.monotonic() - self.last_log > 1:
            self._write_state()
            self.last_log = time.monotonic()

    def close(self):
        self.state = dict(self.state, running=False, ended_at=time.time())
        self.writer.publish(self.state, self.images)
        self._write_state()
        self.writer.close()
        self.log.close()
