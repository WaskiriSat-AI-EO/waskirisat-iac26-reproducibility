"""Deterministic seed derivation for pipeline stages.

Maps a tuple of identifiers (scene_id, tile_idx, profile, role, ...) to a
stable 31-bit integer seed that is independent of Python's salted hash() and
of process/run/iteration order.

Usage::

    from shared.seeding import derive_seed

    seed = derive_seed(global_seed, scene_id, tile_idx, profile, "fpn")
"""

import hashlib


def derive_seed(*parts) -> int:
    """Return a reproducible 31-bit seed from an arbitrary tuple of values.

    The pipe-separated string of str(part) is hashed with blake2b (4 bytes).
    Including the global pipeline seed as the first part ensures that changing
    --seed shifts the entire dataset while per-tile ordering has no effect.

    Args:
        *parts: Any values whose str() representation is stable (int, str, ...).

    Returns:
        Integer in [0, 2**31).
    """
    key = "|".join(str(p) for p in parts).encode()
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int.from_bytes(digest, "big") % (2**31)
