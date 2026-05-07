"""
PrimeLab Core: Sieve Engine
Segmented Sieve of Eratosthenes optimized for large ranges + Miller-Rabin fallback.
"""

import numpy as np
import math
import os
import hashlib
from pathlib import Path
from typing import Generator, List, Optional, Tuple
import multiprocessing as mp
from functools import lru_cache

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Basic sieve (small ranges, ≤ 10^7)
# ─────────────────────────────────────────────────────────────────────────────

def sieve_eratosthenes(n: int) -> np.ndarray:
    """Classic sieve up to n (inclusive). Returns sorted numpy array of primes."""
    if n < 2:
        return np.array([], dtype=np.int64)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0].astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
#  Segmented Sieve (large ranges, ≤ 10^12)
# ─────────────────────────────────────────────────────────────────────────────

def segmented_sieve(low: int, high: int, segment_size: int = 2**19) -> np.ndarray:
    """
    Segmented Sieve of Eratosthenes for range [low, high].
    Memory-efficient: processes in blocks of `segment_size`.
    """
    if high < low or high < 2:
        return np.array([], dtype=np.int64)

    low = max(low, 2)
    sqrt_high = int(math.isqrt(high)) + 1
    small_primes = sieve_eratosthenes(sqrt_high)

    results = []

    for seg_low in range(low, high + 1, segment_size):
        seg_high = min(seg_low + segment_size - 1, high)
        size = seg_high - seg_low + 1
        sieve = np.ones(size, dtype=bool)

        for p in small_primes:
            if p * p > seg_high:
                break
            # First multiple of p >= seg_low
            start = ((seg_low + p - 1) // p) * p
            if start == p:
                start += p
            if start <= seg_high:
                sieve[start - seg_low::p] = False

        if seg_low == 2:
            sieve[0] = False  # 0 is not prime
        if seg_low <= 1 <= seg_high:
            sieve[1 - seg_low] = False

        primes_in_seg = np.where(sieve)[0] + seg_low
        results.append(primes_in_seg.astype(np.int64))

    if results:
        return np.concatenate(results)
    return np.array([], dtype=np.int64)


def segmented_sieve_generator(low: int, high: int,
                               segment_size: int = 2**19) -> Generator[np.ndarray, None, None]:
    """Generator version: yields numpy arrays of primes per segment (streaming)."""
    if high < low or high < 2:
        return
    low = max(low, 2)
    sqrt_high = int(math.isqrt(high)) + 1
    small_primes = sieve_eratosthenes(sqrt_high)

    for seg_low in range(low, high + 1, segment_size):
        seg_high = min(seg_low + segment_size - 1, high)
        size = seg_high - seg_low + 1
        sieve = np.ones(size, dtype=bool)

        for p in small_primes:
            if p * p > seg_high:
                break
            start = ((seg_low + p - 1) // p) * p
            if start == p:
                start += p
            if start <= seg_high:
                sieve[start - seg_low::p] = False

        if seg_low <= 2 <= seg_high:
            sieve[2 - seg_low] = True
        if seg_low <= 1 <= seg_high:
            sieve[1 - seg_low] = False
        if seg_low == 0:
            sieve[0] = False
            if size > 1:
                sieve[1] = False

        primes_in_seg = np.where(sieve)[0] + seg_low
        yield primes_in_seg.astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
#  Miller-Rabin Primality Test (deterministic for n < 3.3 × 10^24)
# ─────────────────────────────────────────────────────────────────────────────

def _miller_rabin_witness(n: int, a: int) -> bool:
    """Returns True if a is a witness to compositeness of n."""
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return False
    for _ in range(r - 1):
        x = pow(x, 2, n)
        if x == n - 1:
            return False
    return True  # composite


# Deterministic witnesses for ranges up to 3.3×10^24
_WITNESSES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def is_prime_miller_rabin(n: int) -> bool:
    """Deterministic Miller-Rabin primality test."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    for a in _WITNESSES:
        if a >= n:
            continue
        if _miller_rabin_witness(n, a):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Parallel sieve worker
# ─────────────────────────────────────────────────────────────────────────────

def _sieve_worker(args: Tuple) -> np.ndarray:
    low, high, small_primes_list = args
    small_primes = np.array(small_primes_list, dtype=np.int64)
    return segmented_sieve(low, high)


def parallel_sieve(low: int, high: int, n_workers: Optional[int] = None) -> np.ndarray:
    """
    Parallel segmented sieve using multiprocessing.
    Splits [low, high] into n_workers chunks.
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    if high - low < 10**6 or n_workers <= 1:
        return segmented_sieve(low, high)

    chunk_size = (high - low + 1) // n_workers
    ranges = []
    for i in range(n_workers):
        chunk_low = low + i * chunk_size
        chunk_high = low + (i + 1) * chunk_size - 1 if i < n_workers - 1 else high
        ranges.append(chunk_low, chunk_high)

    sqrt_high = int(math.isqrt(high)) + 1
    small_primes = sieve_eratosthenes(sqrt_high).tolist()

    args = [(r[0], r[1], small_primes) for r in ranges]

    with mp.Pool(n_workers) as pool:
        results = pool.map(_sieve_worker, args)

    return np.concatenate(results)


# ─────────────────────────────────────────────────────────────────────────────
#  Disk Cache
# ─────────────────────────────────────────────────────────────────────────────

def _cache_key(low: int, high: int) -> str:
    key = f"primes_{low}_{high}"
    return hashlib.md5(key.encode()).hexdigest()


def get_primes_cached(low: int, high: int, force_recompute: bool = False) -> np.ndarray:
    """
    Get primes in [low, high] with disk caching (.npy files).
    """
    ckey = _cache_key(low, high)
    cache_path = CACHE_DIR / f"{ckey}.npy"

    if not force_recompute and cache_path.exists():
        return np.load(str(cache_path))

    primes = segmented_sieve(low, high)
    np.save(str(cache_path), primes)
    return primes


def clear_cache():
    """Remove all cached prime files."""
    for f in CACHE_DIR.glob("*.npy"):
        f.unlink()


def cache_info() -> dict:
    """Return info about cached files."""
    files = list(CACHE_DIR.glob("*.npy"))
    total_size = sum(f.stat().st_size for f in files)
    return {
        "n_files": len(files),
        "total_size_mb": total_size / (1024 * 1024),
        "files": [f.name for f in files],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  High-level API
# ─────────────────────────────────────────────────────────────────────────────

def get_primes(low: int = 2, high: int = 10**6,
               use_cache: bool = True, use_parallel: bool = False) -> np.ndarray:
    """
    Main entry point for prime generation.
    Auto-selects algorithm based on range size.
    """
    low = max(2, low)
    if high < low:
        return np.array([], dtype=np.int64)

    if use_cache:
        return get_primes_cached(low, high)

    span = high - low
    if span > 10**7 and use_parallel:
        return parallel_sieve(low, high)
    else:
        return segmented_sieve(low, high)


def nth_prime_estimate(n: int) -> int:
    """Approximate the n-th prime using inverse prime counting function."""
    if n < 6:
        return [2, 3, 5, 7, 11, 13][n - 1]
    ln_n = math.log(n)
    return int(n * (ln_n + math.log(ln_n)))
