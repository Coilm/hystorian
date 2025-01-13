import numpy as np
import numpy.typing as npt


def extract_hysteresis(
    *chans: list[npt.NDArray],
    len_bias: int,
    waveform_pulsetime: int,
    waveform_dutycycle: float = 0.5,
    num_pts_per_sec: float
) -> tuple[npt.NDArray]:
    """extract_hist split data from an SSPFM measurement into part where the bias is off, and the bias is on.
    Therefore it can be used to build P-E loop.

    Parameters
    ----------
    len_bias : int
        len_bias is the lenghth of the bias applied during the SSPF.
        If the data is from an Asylum ARDF you can find it by, for example, doing f.read("datasets/.../bias/retrace")[0,0,:]
    waveform_pulsetime : int
        The length in time of the pulse.
        If from Asylum can be found in metadata under name "ARDoIVArg3". f.read("metadata/.../ArDoIVArg3")
    NumPtsPerSec : float
        The number of points per seconds in a pulse.
        If from Asylum can be computed from the metadata "NumPtsPerSec": float(f.read("metadata/.../NumPtsPerSec"))
    waveform_dutycycle : float, optional
        The length in time of the pulse.
        If from Asylum can be found in metadata under name "ARDoIVArg4", by default 0.5 (which is the default value of Asylum)

    Returns
    -------
    tuple[npt.NDArray]
        The tuple is twice the size of chans, containing chans[0] 'on', chans[0] 'off', chans[1] 'on', ...
        It splits each channel into on/off where the bias waveform was either 0 (off) or something else (on)
    """

    output = []
    waveform_delta = 1 / num_pts_per_sec
    waveform_numbiaspoints = int(np.floor(waveform_delta * len_bias / waveform_pulsetime))
    waveform_pulsepoints = int(waveform_pulsetime / waveform_delta)
    waveform_offpoints = int(waveform_pulsepoints * (1.0 - waveform_dutycycle))

    for chan in chans:
        result_on = np.ndarray(shape=(np.shape(chan)[0], np.shape(chan)[1], waveform_numbiaspoints))
        result_off = np.ndarray(shape=(np.shape(chan)[0], np.shape(chan)[1], waveform_numbiaspoints))
        for b in range(waveform_numbiaspoints):
            start = b * waveform_pulsepoints + waveform_offpoints
            stop = (b + 1) * waveform_pulsepoints

            var2 = stop - start + 1
            realstart = int(start + var2 * 0.25)
            realstop = int(stop - var2 * 0.25)
            result_on[:, :, b] = np.nanmean(chan[:, :, realstart:realstop], axis=2)
            start = stop
            stop = stop + waveform_pulsepoints * waveform_dutycycle

            var2 = stop - start + 1
            realstart = int(start + var2 * 0.25)
            realstop = int(stop - var2 * 0.25)
            result_off[:, :, b] = np.nanmean(chan[:, :, realstart:realstop], axis=2)
        output.append(result_on)
        output.append(result_off)

    output = tuple(output)

    output = tuple(output)
    return output
