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


def correlation(l: list[float], r: list[float]) -> float:
    """Return the Pearson correlation coefficient of two channels.

    Both channel means are removed before normalisation, so a constant DC
    offset does not affect the result.  The returned value is always in
    ``[-1.0, 1.0]`` within ``1e-6`` and is never ``NaN``.  If either channel
    has zero variance (including the all-silent case), return ``0.0``.

    Raises:
        PhaseError: if the two channels have different lengths.
    """
    raise NotImplementedError("correlation is not implemented")


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
    raise NotImplementedError("detect_dual_mono is not implemented")


def mid_side(l: list[float], r: list[float]) -> tuple[list[float], list[float]]:
    """Split a stereo pair into normalised mid/side channels.

    Returns ``(mid, side)`` where ``mid[i] = (l[i] + r[i]) / 2`` and
    ``side[i] = (l[i] - r[i]) / 2``.  Both channels therefore have unit
    gain (the side channel is not doubled).  An all-zero side channel means
    the source is mono.  Input channels are not mutated.

    Raises:
        PhaseError: if the two channels have different lengths.
    """
    raise NotImplementedError("mid_side is not implemented")


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
    raise NotImplementedError("fix_polarity is not implemented")


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
    raise NotImplementedError("mono_compat is not implemented")


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
    raise NotImplementedError("windowed_correlation is not implemented")


def validate(l: list[float], r: list[float], threshold: float) -> list[str]:
    """Return human-readable issues for a stereo pair and threshold.

    This function reports problems instead of raising for bad data.  It
    returns ``[]`` when the pair is valid, otherwise a list containing the
    applicable messages:

    * ``"length mismatch: left=... right=..."``
    * ``"threshold out of range: must be within [-1, 1]"`
    * ``"all silent: both channels contain only zeros"``

    Raises:
        PhaseError: only for non-numeric input where no useful report can
            be produced.
    """
    raise NotImplementedError("validate is not implemented")


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
    raise NotImplementedError("detect_batch is not implemented")
