"""Deterministic, implementation-independent randomness for the agentic suite.

Everything that must be reproducible across machines and Python versions is
derived from SHA-256 / SHAKE-256 over explicit ASCII labels, never from
`random` or from unseeded library state. Two properties matter:

  * committed: given the label and the committed integer seed, anyone can
    regenerate the exact byte stream;
  * unforgeable (when keyed): tokens minted with a run secret cannot be
    predicted by the model under evaluation, which never sees the secret.
"""

from __future__ import annotations

import hashlib
import hmac
import json


def canonical_json(obj) -> str:
    """Sorted keys, compact separators: the committed serialization."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(obj) -> str:
    """SHA-256 hex of the canonical JSON serialization."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def stream_bytes(seed: int, label: str, n: int) -> bytes:
    """n deterministic bytes from SHAKE-256 over `seed:label`."""
    key = f"{seed:#x}:{label}".encode("ascii")
    return hashlib.shake_256(key).digest(n)


def stream_u64(seed: int, label: str, n: int):
    """n deterministic uint64s as a numpy array (used by the bootstrap)."""
    import numpy as np

    raw = stream_bytes(seed, label, n * 8)
    return np.frombuffer(raw, dtype="<u8")


def choice_indices(seed: int, label: str, n_draws: int, modulus: int):
    """n_draws indices in [0, modulus) with negligible (2^-64-scale) modulo bias."""
    return (stream_u64(seed, label, n_draws) % modulus).astype("int64")


def permutation(seed: int, label: str, n: int) -> list[int]:
    """A deterministic permutation of range(n) via key-sorting the SHAKE stream."""
    keys = stream_u64(seed, label, n)
    import numpy as np

    return [int(i) for i in np.argsort(keys, kind="stable")]


def hmac_token(secret: bytes, label: str, bits: int = 128) -> str:
    """Keyed token, hex-encoded. Unpredictable without `secret`."""
    if bits % 8:
        raise ValueError("bits must be a multiple of 8")
    mac = hmac.new(secret, label.encode("utf-8"), hashlib.sha256).digest()
    return mac[: bits // 8].hex()


# ---------------------------------------------------------------------------
# SHA-256 counter-mode stream (suite v1 task generation)
#
# The generation side needs stateful draws (rejection-sampled integers,
# Fisher-Yates shuffles, variable-length token mints) that the fixed-length
# label streams above are awkward for. CounterRNG is the committed primitive:
# block_n = SHA256(seed_material || n), seed_material = SHA256(labels).
# Same guarantees: no Python `random`, bit-identical on every platform.
# ---------------------------------------------------------------------------

_B32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"


class CounterRNG:
    """Deterministic byte stream with label-derived children.

    Every generator derives a child stream from string labels, so the values
    drawn for task i never depend on how many draws task i-1 consumed.
    """

    def __init__(self, *labels: object) -> None:
        material = "\x1f".join(str(x) for x in labels).encode("utf-8")
        self._seed = hashlib.sha256(material).digest()
        self._counter = 0
        self._buf = b""

    def derive(self, *labels: object) -> "CounterRNG":
        """A child stream keyed by this stream's seed plus extra labels."""
        return CounterRNG(self._seed.hex(), *labels)

    # -- raw bytes ----------------------------------------------------------

    def take(self, n: int) -> bytes:
        while len(self._buf) < n:
            block = hashlib.sha256(
                self._seed + self._counter.to_bytes(8, "big")).digest()
            self._counter += 1
            self._buf += block
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    # -- integers -------------------------------------------------------------

    def randbits(self, k: int) -> int:
        if k <= 0:
            raise ValueError("k must be positive")
        nbytes = (k + 7) // 8
        v = int.from_bytes(self.take(nbytes), "big")
        return v >> (nbytes * 8 - k)

    def randint(self, a: int, b: int) -> int:
        """Uniform integer in [a, b] inclusive, unbiased via rejection sampling."""
        if b < a:
            raise ValueError("empty range")
        n = b - a + 1
        k = max(1, (n - 1).bit_length())
        while True:
            v = self.randbits(k)
            if v < n:
                return a + v

    # -- sequences ------------------------------------------------------------

    def choice(self, seq):
        if not seq:
            raise ValueError("empty sequence")
        return seq[self.randint(0, len(seq) - 1)]

    def shuffle(self, seq) -> list:
        """Fisher-Yates on a copy; the input is never mutated."""
        out = list(seq)
        for i in range(len(out) - 1, 0, -1):
            j = self.randint(0, i)
            out[i], out[j] = out[j], out[i]
        return out

    # -- tokens -----------------------------------------------------------------

    def base32(self, chars: int) -> str:
        """Uppercase RFC-4648 base32 token: 5 unbiased bits per character."""
        return "".join(_B32[self.randbits(5)] for _ in range(chars))

    def hex_token(self, bits: int = 128) -> str:
        if bits % 8:
            raise ValueError("bits must be a multiple of 8")
        return self.take(bits // 8).hex()
