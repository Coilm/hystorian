from unittest.mock import patch

import numpy as np
import pytest

from hystorian.processing.spm import (
    binarize_phase,
    get_phase_unwrapping_shift,
    shift_and_wrap_phase,
)


def test_get_phase_unwrapping_shift_simple():
    phase = np.array([350, 355, 0, 5])
    shift = get_phase_unwrapping_shift(phase, phase_step=90)
    assert isinstance(shift, (int, np.integer))
    assert shift in [0, 90, 180, 270]  # Must be a valid shift step


def test_get_phase_unwrapping_shift_constant_array():
    phase = np.ones(10) * 45
    shift = get_phase_unwrapping_shift(phase, phase_step=90)
    assert shift % 90 == 0


def test_shift_and_wrap_phase_basic():
    phase = np.array([350, 355, 0, 5])
    shifted = shift_and_wrap_phase(phase, phase_step=90)
    assert shifted.shape == phase.shape
    assert np.all((shifted >= 0) & (shifted < 360))


def test_shift_and_wrap_phase_no_jump_needed():
    phase = np.array([10, 20, 30, 40])
    shifted = shift_and_wrap_phase(phase, phase_step=90)
    assert np.allclose(np.sort(shifted), np.sort(phase % 360))


@patch("hystorian.processing.spm.get_phase_unwrapping_shift", return_value=0)
def test_binarize_phase_two_peaks(mock_shift):
    np.random.seed(0)  # for reproducibility
    peak1 = np.random.normal(30, 2, size=500)
    peak2 = np.random.normal(200, 2, size=500)
    phase = np.concatenate([peak1, peak2]) % 360
    binary = binarize_phase(phase)
    assert set(np.unique(binary)) == {0, 1}
    assert binary.shape == phase.shape


@patch("hystorian.processing.spm.get_phase_unwrapping_shift", return_value=0)
def test_binarize_phase_all_same_value(mock_shift):
    phase = np.ones(100) * 50
    binary = binarize_phase(phase)
    assert np.all((binary == 0) | (binary == 1))
