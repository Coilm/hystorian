import numpy as np
import numpy.typing as npt
from .operation import gauss_area


def pfm_params_map(bias: npt.NDArray, phase: npt.NDArray) -> tuple[npt.NDArray]:
    """PFM_params_map calculates physically relevant hysteresis parameters from bias and phase channels.

    Parameters
    ----------
    bias : npt.NDArray
        array containing the bias acquired during an SSPFM. This should be the "on" bias computed with extract_hist
    phase : npt.NDArray
        array containing the phase acquired during an SSPFM. This should be the "off" phase computed with extract_hist

    Returns
    -------
    coerc_pos: npt.NDArray
        the positive coercive bias
    coerc_neg: npt.NDArray
        the negative coercive bias
    step_left: npt.NDArray
        The size of the phase jump (left side)
    step_right: npt.NDArray
        The size of the phase jump (right side)
    imprint: npt.NDArray
        The imprint of the switching loop
    phase_shift: npt.NDArray
        The phase shift of the switching loop

    """
    x, y, _ = np.shape(phase)
    coerc_pos = np.zeros((x, y), dtype=float)
    coerc_neg = np.zeros((x, y), dtype=float)
    step_left = np.zeros((x, y), dtype=float)
    step_right = np.zeros((x, y), dtype=float)
    imprint = np.zeros((x, y), dtype=float)
    phase_shift = np.zeros((x, y), dtype=float)
    for xi in range(x):
        for yi in range(y):
            hyst_matrix = _calc_hyst_params(bias[xi, yi, :], phase[xi, yi, :])
            coerc_pos[xi, yi] = hyst_matrix[0]
            coerc_neg[xi, yi] = hyst_matrix[1]
            step_left[xi, yi] = hyst_matrix[2]
            step_right[xi, yi] = hyst_matrix[3]
            imprint[xi, yi] = (hyst_matrix[0] + hyst_matrix[1]) / 2.0
            phase_shift[xi, yi] = hyst_matrix[3] - hyst_matrix[2]

    return coerc_pos, coerc_neg, step_left, step_right, imprint, phase_shift


def _calc_hyst_params(bias: npt.NDArray, phase: npt.NDArray) -> list[npt.NDArray]:
    """
    Calculate hysteresis parameters from bias and phase channels.
    Used in PFM_params_map. Would recommend using the function instead.

    Parameters
    ----------
    bias : npt.NDArray
        array containing the bias acquired during an SSPFM. This should be the "on" bias computed with extract_hist
    phase :npt.NDArray
        array containing the phase acquired during an SSPFM. This should be the "off" phase computed with extract_hist

    Returns
    -------
        list[npt.NDArray]
        list where the elements correspond to :
            - The positive coercive bias
            - The negative coercive bias
            - The size of the phase jump (left side)
            - The size of the phase jump (right side)

    """
    biasdiff = np.diff(bias)
    up = np.sort(np.unique(np.hstack((np.where(biasdiff > 0)[0], np.where(biasdiff > 0)[0] + 1))))
    dn = np.sort(np.unique(np.hstack((np.where(biasdiff < 0)[0], np.where(biasdiff < 0)[0] + 1))))
    phase_shift = get_phase_unwrapping_shift(phase)

    # UP leg calculations
    if up.size == 0:
        step_left_up = np.nan
        step_right_up = np.nan
        coercive_volt_up = np.nan
    else:
        x = np.array(bias[up])
        y = (np.array(phase[up]) + phase_shift) % 360
        step_left_up = np.median(y[np.where(x == np.min(x))[0]])
        step_right_up = np.median(y[np.where(x == np.max(x))[0]])

        avg_x = []
        avg_y = []
        for v in np.unique(x):
            avg_x.append(v)
            avg_y.append(np.mean(y[np.where(x == v)[0]]))

        my_x = np.array(avg_x)[1:]
        my_y = np.abs(np.diff(avg_y))

        coercive_volt_up = my_x[np.nanargmax(my_y)]

    # DOWN leg calculations
    if dn.size == 0:
        step_left_dn = np.nan
        step_right_dn = np.nan
        coercive_volt_dn = np.nan
    else:
        x = np.array(bias[dn])
        y = (np.array(phase[dn]) + phase_shift) % 360
        step_left_dn = np.median(y[np.where(x == np.min(x))[0]])
        step_right_dn = np.median(y[np.where(x == np.max(x))[0]])

        avg_x = []
        avg_y = []
        for v in np.unique(x):
            avg_x.append(v)
            avg_y.append(np.mean(y[np.where(x == v)[0]]))

        my_x = np.array(avg_x)[1:]
        my_y = np.abs(np.diff(avg_y))

        coercive_volt_dn = my_x[np.nanargmax(my_y)]

    return [
        coercive_volt_up,
        coercive_volt_dn,
        np.nanmean([step_left_dn, step_left_up]),
        np.nanmean([step_right_dn, step_right_up]),
    ]


def clean_loop(bias, phase, amp, threshold=None):
    """
    Used to determine if a SSPFM loop is good or not by calculating the area encompassed by the
    hysteresis curve and comparing it to a threshold

    Parameters
    ----------
    bias: nd-array
        3d-array containing the bias applied to each point of the grid
    phase: nd-array
        3d-array containing the phase applied to each point of the grid
    amp: nd-array
        3d-array containing the amplitude applied to each point of the grid
    threshold: int or float, optional
        minimal value of the loop area to be considered a good loop (default: None)
        if set to None will use threshold = np.mean(area_grid_full) - 2 * np.std(area_grid_full)

    Returns
    -------
        good_bias: list
            list of the bias corresponding to good loops
        good_phase: list
            list of the phase corresponding to good loops
        good_amp: list
            list of the amplitudes corresponding to good loops
        mask: list
            a 2D mask where a 1 correspond to a good loop and 0 to a bad loop, can be used to mask
            the input data.
    """
    good_bias = []
    good_phase = []
    good_amp = []

    mask = np.ndarray((np.shape(bias)[0], np.shape(bias)[1]))

    if threshold is None:
        area_grid_full = np.ndarray((np.shape(bias)[0], np.shape(bias)[1]))
        for xi in range(np.shape(bias)[0]):
            for yi in range(np.shape(bias)[1]):
                area_grid_full[xi, yi] = gauss_area(bias[xi, yi, :], phase[xi, yi, :])
            threshold = np.mean(area_grid_full) - 2 * np.std(area_grid_full)

    for xi in range(np.shape(bias)[0]):
        for yi in range(np.shape(bias)[1]):
            if gauss_area(bias[xi, yi, :], phase[xi, yi, :]) > threshold:
                good_bias.append(bias[xi, yi, :])
                good_phase.append(phase[xi, yi, :])
                good_amp.append(amp[xi, yi, :])
                mask[xi, yi] = True
            else:
                mask[xi, yi] = False

    return good_bias, good_phase, good_amp, mask


def get_phase_unwrapping_shift(phase, phase_step=1):
    """
    Finds the smallest shift that minimizes the phase jump in a phase series

    Parameters
    ----------
    phase : a 1D array of consecutive phase measurements
    phase_step : the algorithm will try shifts every phase_step. Increase this to speed up execution.

    Returns
    -------
    The phase shift. (phase + shift) % 360 will have the smallest jumps between consecutive phase points.
    """
    phase_step = 90
    jumps = []
    for shift in range(0, 360, phase_step):
        y = (phase + shift) % 360
        # what is the biggest phase jump?
        max_phase_jump = np.max(np.abs(np.diff(y)))
        jumps.append(max_phase_jump)

    return phase_step * np.argmin(np.array(jumps))
