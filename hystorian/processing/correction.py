import itertools
import warnings

import numpy as np


def line_fit(line, order=1, box=None):
    """
    Do a nth order polynomial line flattening

    Parameters
    ----------
    line : 1d array-like
    order : integer

    Returns
    -------
    result : 1d array-like
        same shape as data
    """
    if order < 0:
        raise ValueError("expected deg >= 0")
    newline = line
    if box:
        if len(box) == 2:
            newline = line[box[0] : box[1]]
        else:
            raise ValueError("Box should be a 2-length list so that the list is cutted has line[box[0], box[1]].")
    x = np.arange(len(newline))
    k = np.isfinite((newline))
    if not np.isfinite(newline).any():
        warnings.warn("The line does not contain any finite values, returning the unmodified line.")
        return line

    coefficients = np.polyfit(x[k], newline[k], order)
    return line - np.polyval(coefficients, np.arange(len(line)))


def line_flatten_image(data, order=1, axis=0, box=None):
    """
    Do a line flattening

    Parameters
    ----------
    data : 2d array
    order : integer
    axis : integer
        axis perpendicular to lines

    Returns
    -------
    result : array-like
        same shape as data
    """

    if axis == 1:
        data = data.T

    ndata = np.zeros_like(data)

    for i, line in enumerate(data):
        ndata[i, :] = line_fit(line, order, box)

    if axis == 1:
        ndata = ndata.T

    return ndata


def plane_flatten_image(data, order=1, box=None):
    """
    Do a plane flattening

    Parameters
    ----------
    data : 2d array
    order : integer

    Returns
    -------
    result : array-like
        same shape as data
    """
    if box is None:
        box = []

    fitdata = data
    if len(box) == 4:
        fitdata = data[box[0] : box[1], box[2] : box[3]]
    xx, yy = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    xxfit, yyfit = np.meshgrid(np.arange(fitdata.shape[1]), np.arange(fitdata.shape[0]))
    m = _polyfit2d(xxfit.ravel(), yyfit.ravel(), fitdata.ravel(), order=order)
    return data - _polyval2d(xx, yy, m)


def _polyfit2d(x, y, z, order=1):
    """
    Fit a 2D polynomial model to a dataset

    Parameters
    ----------
    x : list or array
    y : list or array
    z : list or array
    order : int

    Returns
    -------
    m : array-like
        list of indexes
    """
    ncols = (order + 1) ** 2
    g = np.zeros((x.size, ncols))
    ij = itertools.product(range(order + 1), range(order + 1))
    for k, (i, j) in enumerate(ij):
        g[:, k] = x**i * y**j
    k = np.isfinite(z)
    m, _, _, _ = np.linalg.lstsq(g[k], z[k], rcond=None)
    return m


def _polyval2d(x, y, m):
    """
    Applies polynomial indices obtained from polyfit2d on an x and y

    Parameters
    ----------
    x : list or array
    y : list or array
    m : list
        polynomials used

    Returns
    -------
    z : array-like
        array of heights
    """
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order + 1), range(order + 1))
    z = np.zeros_like(x, dtype=float)
    for a, (i, j) in zip(m, ij):
        z += a * x**i * y**j
    return z
