"""Core primitives for dual-mono and polarity checks.

All correlation-style values are in ``[-1.0, 1.0]`` and are compared in
tests with an absolute tolerance of ``1e-6``.  A silent channel pair has a
correlation of ``0.0`` (never ``NaN``).  Channel pairs are always passed as
two equal-length ``list[float]`` objects: ``(left, right)``.
"""

from __future__ import annotations

import math


class PhaseError(ValueError):
    """Raised when channel data or detection parameters are invalid."""


def _check_lengths(l: list[float], r: list[float]) -> None:
    if len(l) != len(r):
        raise PhaseError(
            f"length mismatch: left={len(l)} right={len(r)}"
        )


def _check_threshold(threshold: float) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise PhaseError(f"threshold must be a number, got {type(threshold).__name__}")
    if not -1.0 <= threshold <= 1.0:
        raise PhaseError("threshold out of range: must be within [-1, 1]")


def correlation(l: list[float], r: list[float]) -> float:
    """Return the Pearson correlation coefficient of two channels.

    Both channel means are removed before normalisation, so a constant DC
    offset does not affect the result.  The returned value is always in
    ``[-1.0, 1.0]`` within ``1e-6`` and is never ``NaN``.  If either channel
    has zero variance (including the all-silent case), return ``0.0``.

    Raises:
        PhaseError: if the two channels have different lengths.
    """
    _check_lengths(l, r)
    n = len(l)
    if n == 0:
        return 0.0
    mean_l = sum(l) / n
    mean_r = sum(r) / n
    num = 0.0
    var_l = 0.0
    var_r = 0.0
    for a, b in zip(l, r):
        da = a - mean_l
        db = b - mean_r
        num += da * db
        var_l += da * da
        var_r += db * db
    if var_l == 0.0 or var_r == 0.0:
        # Silent (or constant) channel: no zero-division, defined as 0.0.
        return 0.0
    c = num / math.sqrt(var_l * var_r)
    # Clamp tiny floating-point overshoot so the result stays in [-1, 1].
    return max(-1.0, min(1.0, c))


def detect_dual_mono(l: list[float], r: list[float], threshold: float) -> bool:
    """Return whether a stereo pair is dual mono with inverted polarity.

    A pair is dual mono only when it is both highly correlated and
    anti-phase, i.e. ``correlation(l, r) <= -abs(threshold)``.  High
    in-phase correlation alone must not be reported as dual mono.
    ``threshold`` is a correlation magnitude in ``[-1.0, 1.0]`` (normally
    ``0.9``).

    Raises:
        PhaseError: for mismatched lengths or a threshold outside
            ``[-1.0, 1.0]``.
    """
    _check_threshold(threshold)
    _check_lengths(l, r)
    return correlation(l, r) <= -abs(threshold)


def mid_side(l: list[float], r: list[float]) -> tuple[list[float], list[float]]:
    """Split a stereo pair into normalised mid/side channels.

    Returns ``(mid, side)`` where ``mid[i] = (l[i] + r[i]) / 2`` and
    ``side[i] = (l[i] - r[i]) / 2``.  Both channels therefore have unit
    gain (the side channel is not doubled).  An all-zero side channel means
    the source is mono.  Input channels are not mutated.

    Raises:
        PhaseError: if the two channels have different lengths.
    """
    _check_lengths(l, r)
    mid = [(a + b) / 2.0 for a, b in zip(l, r)]
    side = [(a - b) / 2.0 for a, b in zip(l, r)]
    return mid, side


def fix_polarity(
    l: list[float], r: list[float], invert: bool
) -> tuple[list[float], list[float]]:
    """Repair an inverted pair by optionally negating only the right channel.

    If ``invert`` is true, return copies of ``l`` and ``r`` with every
    ``right`` sample negated; the left channel is never changed.  If
    ``invert`` is false, return unchanged copies.  Inputs are not mutated.

    Raises:
        PhaseError: if the two channels have different lengths.
    """
    _check_lengths(l, r)
    if invert:
        return list(l), [-x for x in r]
    return list(l), list(r)


def mono_compat(l: list[float], r: list[float]) -> float:
    """Return mono fold-down loss in dB.

    The fold-down signal is ``m[i] = (l[i] + r[i]) / 2``.  With
    ``rms_ref = sqrt((rms(l) ** 2 + rms(r) ** 2) / 2)``, the result is
    ``20 * log10(rms(m) / rms_ref)``.  Identical channels therefore give
    ``0.0`` dB, while a non-silent perfectly anti-phase pair folds to
    silence and gives ``float("-inf")``.  The result must be a dB value,
    not a raw linear ratio.

    Raises:
        PhaseError: if the two channels have different lengths.
    """
    _check_lengths(l, r)
    n = len(l)
    if n == 0:
        return 0.0
    sum_l = sum(x * x for x in l)
    sum_r = sum(x * x for x in r)
    sum_m = sum(((a + b) / 2.0) ** 2 for a, b in zip(l, r))
    rms_ref = math.sqrt((sum_l / n + sum_r / n) / 2.0)
    rms_m = math.sqrt(sum_m / n)
    if rms_ref == 0.0:
        # Fully silent pair: nothing is lost by folding down.
        return 0.0
    if rms_m == 0.0:
        return float("-inf")
    return 20.0 * math.log10(rms_m / rms_ref)


def windowed_correlation(
    l: list[float], r: list[float], window: int, hop: int
) -> list[float]:
    """Return correlations for overlapping windows of the stereo pair.

    Windows have length ``window`` and start at ``0, hop, 2 * hop, ...``
    while ``start + window <= len(l)``.  This means ``hop < window``
    produces overlap; an inversion crossing a non-overlapping boundary must
    still be found.  Each element is the Pearson correlation of that slice
    (same contract as :func:`correlation`), and silent windows contribute
    ``0.0``.  If there are no complete windows, return ``[]``.  Inputs are
    not mutated.

    Raises:
        PhaseError: for mismatched lengths, or if ``window`` or ``hop`` is
            not a positive integer.
    """
    _check_lengths(l, r)
    for name, value in (("window", window), ("hop", hop)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PhaseError(f"{name} must be a positive integer, got {value!r}")
    results: list[float] = []
    start = 0
    while start + window <= len(l):
        results.append(correlation(l[start:start + window], r[start:start + window]))
        start += hop
    return results


def validate(l: list[float], r: list[float], threshold: float) -> list[str]:
    """Return human-readable issues for a stereo pair and threshold.

    This function reports problems instead of raising for bad data.  It
    returns ``[]`` when the pair is valid, otherwise a list containing the
    applicable messages:

    * ``"length mismatch: left=... right=..."``
    * ``"threshold out of range: must be within [-1, 1]"``
    * ``"all silent: both channels contain only zeros"``

    Raises:
        PhaseError: only for non-numeric input where no useful report can
            be produced.
    """
    for name, ch in (("left", l), ("right", r)):
        for x in ch:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise PhaseError(
                    f"non-numeric sample in {name} channel: {x!r}"
                )
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise PhaseError(f"threshold must be a number, got {type(threshold).__name__}")
    issues: list[str] = []
    if len(l) != len(r):
        issues.append(f"length mismatch: left={len(l)} right={len(r)}")
    if not -1.0 <= threshold <= 1.0:
        issues.append("threshold out of range: must be within [-1, 1]")
    if l and r and all(x == 0 for x in l) and all(x == 0 for x in r):
        issues.append("all silent: both channels contain only zeros")
    return issues


def detect_batch(pairs: list[tuple[list[float], list[float]]], config: dict) -> dict:
    """Detect dual-mono polarity faults for many stereo pairs atomically.

    ``config`` is a mapping with a required ``"threshold"`` key and optional
    ``"window"`` / ``"hop"`` keys.  If a window is configured, each pair is
    judged on its overlapping windows; otherwise :func:`detect_dual_mono`
    is used.  Every pair must use independent buffers, so one pair can
    never contaminate the next.

    On success return ``{"ok": True, "results": [bool, ...], "issues": []}``
    with results in the same order as ``pairs``.  If any pair is invalid,
    the whole batch rolls back: return
    ``{"ok": False, "results": [], "issues": [...]}`` and no partial
    results.  Inputs are not mutated.

    Raises:
        PhaseError: for a missing or invalid ``config`` value, or a
            non-iterable ``pairs`` argument.
    """
    if not isinstance(config, dict):
        raise PhaseError("config must be a mapping")
    if "threshold" not in config:
        raise PhaseError("config is missing required key 'threshold'")
    threshold = config["threshold"]
    _check_threshold(threshold)

    window = config.get("window")
    hop = config.get("hop")
    if window is not None or hop is not None:
        if window is None or hop is None:
            raise PhaseError("config must define both 'window' and 'hop' or neither")
        for name, value in (("window", window), ("hop", hop)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PhaseError(f"{name} must be a positive integer, got {value!r}")

    if isinstance(pairs, (str, bytes)) or not hasattr(pairs, "__iter__"):
        raise PhaseError("pairs must be an iterable of (left, right) tuples")
    pairs = list(pairs)

    # Validate every pair first; any invalid pair rolls back the whole batch.
    issues: list[str] = []
    for i, pair in enumerate(pairs):
        try:
            l, r = pair
        except (TypeError, ValueError) as exc:
            raise PhaseError(f"pair {i} is not a (left, right) tuple") from exc
        issues.extend(f"pair {i}: {msg}" for msg in validate(l, r, threshold)
                      if "all silent" not in msg)
    if issues:
        return {"ok": False, "results": [], "issues": issues}

    # Each pair is analysed independently: fresh slices/buffers per pair, so
    # no state can leak from one pair into the next.
    results: list[bool] = []
    for l, r in pairs:
        if window is not None:
            coeffs = windowed_correlation(list(l), list(r), window, hop)
            results.append(any(c <= -abs(threshold) for c in coeffs))
        else:
            results.append(detect_dual_mono(list(l), list(r), threshold))
    return {"ok": True, "results": results, "issues": []}
