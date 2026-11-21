"""Regression tests for dual-mono / polarity detection."""

import math

import pytest

from phase.correlation import (
    PhaseError,
    correlation,
    detect_batch,
    detect_dual_mono,
    fix_polarity,
    mid_side,
    mono_compat,
    validate,
    windowed_correlation,
)

TOL = 1e-6


# --- One test per documented defect -----------------------------------------


def test_correlation_removes_dc_offset():
    left = [100.0, 101.0, 103.0, 106.0, 110.0]
    right = [-50.0, -48.0, -44.0, -38.0, -30.0]  # positive affine map of left
    assert correlation(left, right) == pytest.approx(1.0, abs=TOL)


def test_windowed_correlation_silence_has_no_zero_division():
    silence = [0.0] * 8
    result = windowed_correlation(silence, silence, 4, 2)
    assert result == pytest.approx([0.0, 0.0, 0.0], abs=TOL)


def test_detect_dual_mono_requires_inversion():
    left = [1.0, -2.0, 3.0, -4.0]
    assert detect_dual_mono(left, list(left), 0.9) is False


def test_mid_side_keeps_side_gain_normalized():
    left = [1.0, -2.0, 3.0]
    right = [-1.0, 2.0, -3.0]
    mid, side = mid_side(left, right)
    assert mid == pytest.approx([0.0, 0.0, 0.0], abs=TOL)
    assert side == pytest.approx(left, abs=TOL)


def test_fix_polarity_inverts_right_channel_only():
    left = [1.0, 2.0, 3.0]
    right = [4.0, 5.0, 6.0]
    fixed_left, fixed_right = fix_polarity(left, right, True)
    assert fixed_left == pytest.approx(left, abs=TOL)
    assert fixed_right == pytest.approx([-4.0, -5.0, -6.0], abs=TOL)


def test_mono_compat_reports_db():
    left = [1.0, -1.0, 1.0, -1.0]
    assert mono_compat(left, left) == pytest.approx(0.0, abs=TOL)
    anti = mono_compat(left, [-x for x in left])
    assert math.isinf(anti)
    assert anti < 0.0


def test_windowed_correlation_uses_overlap():
    left = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    right = [1.0, 2.0, -3.0, -4.0, -5.0, -6.0, 7.0, 8.0]
    result = windowed_correlation(left, right, 4, 2)
    assert len(result) == 3
    assert result[1] == pytest.approx(-1.0, abs=TOL)


def test_validate_reports_length_mismatch():
    issues = validate([1.0, 2.0, 3.0], [1.0, 2.0], 0.9)
    assert issues
    assert any("length" in issue.lower() for issue in issues)


def test_detect_batch_does_not_share_buffers():
    config = {"threshold": 0.9, "window": 4, "hop": 2}
    pairs = [
        (
            [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0],
            [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0],
        ),
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ]
    result = detect_batch(pairs, config)
    assert result["ok"] is True
    assert result["results"] == [True, False]


def test_validate_rejects_out_of_range_threshold():
    for bad_threshold in (1.5, -1.5):
        issues = validate([1.0, 2.0], [1.0, 2.0], bad_threshold)
        assert any("threshold" in issue.lower() for issue in issues)


# --- Five combination / contract tests --------------------------------------


def test_identical_channels_correlation_is_one():
    left = [1.0, -2.0, 3.0, 0.5]
    assert correlation(left, list(left)) == pytest.approx(1.0, abs=TOL)


def test_inverted_channels_correlation_is_minus_one():
    left = [1.0, -2.0, 3.0, 0.5]
    right = [-x for x in left]
    assert correlation(left, right) == pytest.approx(-1.0, abs=TOL)


def test_all_silence_correlation_is_zero():
    assert correlation([0.0] * 5, [0.0] * 5) == pytest.approx(0.0, abs=TOL)


def test_correlation_rejects_length_mismatch():
    with pytest.raises(PhaseError):
        correlation([1.0, 2.0, 3.0], [1.0, 2.0])


def test_detect_batch_preserves_success_order():
    pairs = [
        ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]),
        ([1.0, -2.0, 3.0, -4.0], [-1.0, 2.0, -3.0, 4.0]),
    ]
    result = detect_batch(pairs, {"threshold": 0.9})
    assert result["ok"] is True
    assert result["results"] == [False, True]
