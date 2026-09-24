"""Length-prefixed message framing between the closed-loop agent (envs/carla, Python 3.8, NumPy < 1.25)
and a policy server (model venv, Python 3.12, NumPy 2). Pickle is not used on purpose: NumPy 2 pickles
reference `numpy._core`, which NumPy 1.x cannot load.

A message is  <uint32 header_len><header JSON><raw array buffers in header order>.
The header is {"meta": {...}, "arrays": [[name, dtype, shape], ...]}.
"""
import json
import struct

import numpy as np


def _recv_exactly(sock, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if not k:
            raise ConnectionError("peer closed the connection")
        got += k
    return buf


def send(sock, meta, arrays=None):
    arrays = {k: np.ascontiguousarray(v) for k, v in (arrays or {}).items()}
    header = json.dumps({"meta": meta, "arrays": [[k, v.dtype.str, list(v.shape)] for k, v in arrays.items()]},
                        allow_nan=False).encode()
    sock.sendall(struct.pack("<I", len(header)) + header)
    for v in arrays.values():
        sock.sendall(memoryview(v).cast("B"))


def recv(sock):
    (n,) = struct.unpack("<I", bytes(_recv_exactly(sock, 4)))
    header = json.loads(bytes(_recv_exactly(sock, n)))
    arrays = {}
    for name, dtype, shape in header["arrays"]:
        dt = np.dtype(dtype)
        size = int(np.prod(shape)) * dt.itemsize
        arrays[name] = np.frombuffer(_recv_exactly(sock, size), dtype=dt).reshape(shape)
    return header["meta"], arrays
