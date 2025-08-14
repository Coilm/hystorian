import numpy as np
import scipy.ndimage as ndi

from skimage.color import rgb2gray
from skimage.data import astronaut
from hystorian.processing.distortion import find_transform
from skimage.transform import PolynomialTransform, warp

max_error = 4

def compute_error(ir, forward, mat):
    shape = ir.shape
    yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    coords_recovered = forward(mat(coords))
    coords_recovered = coords_recovered[:, [1, 0]]
    # Compute error
    error = np.linalg.norm(coords - coords_recovered, axis=1)
    return np.mean(error)


def test_find_transform_orb_order1():
    ir = rgb2gray(astronaut())[::2, ::2]
    
    params = np.array([
        [5, 1, 0.1, 0, 0, 0],  # x' coefficients
        [-7, 0.1, 1, 0, 0, 0]  # y' coefficients
    ])
    forward  = PolynomialTransform(params, dimensionality=2)
    
    iw = warp(ir, forward, order=1)
    mat = find_transform(ir, iw, method='ORB', order=1)
    error = compute_error(ir, forward, mat)

    assert (
        error < max_error
    ), f"TRE ({error:.2f}) is more than {max_error} pixels."


def test_find_transform_orb_order2():
    ir = rgb2gray(astronaut())[::2, ::2]

    # Second-order polynomial transform coefficients:
    # x' = a0 + a1*x + a2*y + a3*x^2 + a4*x*y + a5*y^2
    # y' = b0 + b1*x + b2*y + b3*x^2 + b4*x*y + b5*y^2
    params = np.array([
        [5, 1, 0.1, 5e-4, 5e-5, 5e-4],  # x' coefficients
        [-7, 0.1, 1, 5e-4, 5e-5, 5e-4]  # y' coefficients
    ])


    forward = PolynomialTransform(params, dimensionality=2)

    # forward  = PolynomialTransform(np.array([[tx, 1, 0], [ty, 0, 1]]), dimensionality=2)
    iw = warp(ir, forward, order=1)
    mat = find_transform(ir, iw, method='ORB', order=1)
    error = compute_error(ir, forward, mat)
    assert (
        error < 2 * max_error # Allowing for some error due to the order of the polynomial
    ), f"TRE ({error:.2f}) is more than {max_error} pixels."