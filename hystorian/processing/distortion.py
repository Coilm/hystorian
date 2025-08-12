import warnings

import numpy as np
from scipy import ndimage as ndi

# This implementation is based on:
# Evangelidis, G. D., & Psarakis, E. Z. (2008). Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization.
# IEEE Transactions on Pattern Analysis and Machine Intelligence, 30(10), 1858–1865.

def custom_warp(im, mat, motion_type="affine", order=1):
    """
    Applies a geometric transformation to an image using a specified transformation matrix.

    Parameters
    ----------
    im : ndarray
        The input image array to be transformed.
    mat : ndarray
        The transformation matrix. For affine, this is typically a 2x3 or 3x3 matrix.
        For homography, this should be a 3x3 matrix.
    motion_type : str, optional
        The type of transformation to apply. Supported values are "affine" and "homography".
        Default is "affine".
    order : int, optional
        The order of the spline interpolation used for the transformation.
        Default is 1 (linear interpolation).

    Returns
    -------
    ndarray
        The transformed (warped) image.

    Notes
    -----
    - This function supports both affine and homography transformations.
    - For affine transformations, it uses `scipy.ndimage.affine_transform`, which applies a linear mapping from input coordinates to output coordinates.
    - For homography transformations, it uses `scipy.ndimage.geometric_transform` with a custom coordinate mapping function, allowing for more complex projective transformations.
    - The function assumes the transformation matrix is correctly formatted for the chosen motion type.
    """
    def coord_mapping(pt, mat):
        pt += (1,)
        points_unwarping = mat @ np.array(pt).T
        return tuple(points_unwarping)

    if motion_type == 'homography':
        return ndi.geometric_transform(
            im, coord_mapping, order=order, extra_arguments=(mat,)
        )
    else:
        return ndi.affine_transform(im, mat, order=order)


def find_transform_ecc(
    ir,
    iw,
    warp_matrix=None,
    motion_type="affine",
    number_of_iterations=200,
    termination_eps=-1.0,
    gauss_filt_size=5.0,
    order=1,
):
    """
    Estimates the geometric transformation matrix that best aligns a warped image (iw) to a reference image (ir)
    by maximizing the Enhanced Correlation Coefficient (ECC).

    The function iteratively updates the transformation matrix to maximize the normalized correlation between
    the reference and warped images. It supports several motion models, including translation, affine, euclidean,
    and homography. Gaussian filtering is applied to both images before alignment to reduce noise and improve stability.

    Parameters
    ----------
    ir : ndarray
        Reference image (target for alignment).
    iw : ndarray
        Warped image to be corrected.
    warp_matrix : ndarray, optional
        Initial guess for the transformation matrix. If None, an identity matrix is used.
        Should be 3x3 for 2D images or 4x4 for 3D images.
        Default is None.
    motion_type : str, optional
        Type of transformation: "translation", "affine", "euclidean", or "homography".
        Default is "affine".
    number_of_iterations : int, optional
        Maximum number of iterations before the algorithm stops. Default is 200.
    termination_eps : float, optional
        Threshold for the absolute difference between the normalized correlation of successive iterations.
        If the difference is less than this value, the algorithm stops. Default is -1.0 (no early stopping).
    gauss_filt_size : float, optional
        Standard deviation for Gaussian kernel used to blur ir and iw before alignment. Default is 5.0.
    order : int, optional
        Interpolation order used in the warp function. Default is 1.

    Returns
    -------
    warp_matrix : ndarray
        The transformation matrix that best aligns iw to ir.

    Raises
    ------
    ValueError
        - If the correlation coefficient (rho) becomes NaN, indicating a failure in the algorithm.
        - If the algorithm stops before convergence, indicating that the images may be uncorrelated or non-overlapping.
    """

    if warp_matrix is None:
        if len(ir.shape) == 2:
            warp_matrix = np.eye(3)
        else:
            warp_matrix = np.eye(4)

    mesh = np.meshgrid(*[np.arange(0, x) for x in ir.shape], indexing='ij')
    mesh = [x.astype(np.float32) for x in mesh]

    ir = ndi.gaussian_filter(ir, gauss_filt_size)
    iw = ndi.gaussian_filter(iw, gauss_filt_size)

    grad = np.gradient(iw)
    rho = -1
    last_rho = -termination_eps

    ir_mean = np.mean(ir)
    ir_std = np.std(ir)
    ir_meancorr = ir - ir_mean

    ir_norm = np.sqrt(np.sum(np.prod(ir.shape)) * ir_std**2)

    for _ in range(number_of_iterations):
        if np.abs(rho - last_rho) < termination_eps:
            break

        iw_warped = custom_warp(iw, warp_matrix, motion_type=motion_type, order=order)

        iw_mean = np.mean(iw_warped[iw_warped != 0])
        iw_std = np.std(iw_warped[iw_warped != 0])
        iw_norm = np.sqrt(np.sum(iw_warped != 0) * iw_std**2)

        iw_warped_meancorr = iw_warped - iw_mean
        grad_iw_warped = np.array(
            [
                custom_warp(g, warp_matrix, motion_type=motion_type, order=order)
                for g in grad
            ]
        )

        jacobian = compute_jacobian(grad_iw_warped, mesh, warp_matrix, motion_type)
        hessian = compute_hessian(jacobian)
        hessian_inv = np.linalg.inv(hessian)

        correlation = np.vdot(ir_meancorr, iw_warped_meancorr)
        last_rho = rho
        rho = correlation / (ir_norm * iw_norm)

        if np.isnan(rho):
            raise ValueError("NaN encoutered.")

        iw_projection = project_onto_jacobian(jacobian, iw_warped_meancorr)
        ir_projection = project_onto_jacobian(jacobian, ir_meancorr)

        iw_hessian_projection = np.matmul(hessian_inv, iw_projection)

        num = (iw_norm**2) - np.dot(iw_projection, iw_hessian_projection)
        den = correlation - np.dot(ir_projection, iw_hessian_projection)
        if den <= 0:
            warnings.warn(
                (
                    "The algorithm stopped before its convergence. The correlation is going to be minimized."
                    "Images may be uncorrelated or non-overlapped."
                ),
                RuntimeWarning,
            )
            return warp_matrix

        _lambda = num / den

        error = _lambda * ir_meancorr - iw_warped_meancorr
        error_projection = project_onto_jacobian(jacobian, error)
        delta_p = np.matmul(hessian_inv, error_projection)
        warp_matrix = update_warping_matrix(warp_matrix, delta_p, motion_type)

    return warp_matrix


def compute_jacobian(grad, xy_grid, warp_matrix, motion_type="affine"):
    """
    Computes the Jacobian matrix of the warped image with respect to the transformation parameters.

    The Jacobian describes how small changes in the transformation parameters affect the pixel intensities
    of the warped image. Mathematically, the Jacobian is defined as the matrix of partial derivatives of the
    warped image with respect to the parameter vector, i.e., J(x; p) = ∂w(x; p)/∂p
    (see Evangelidis & Psarakis, 2008, IEEE TPAMI, p. 1859).

    This function supports multiple motion models, including translation, affine,
    euclidean, homography, and their 3D variants. The appropriate Jacobian computation is selected based
    on the dimensionality of the input and the specified motion type.

    Parameters
    ----------
    grad : ndarray
        Gradient(s) of the warped image.
    xy_grid : list of ndarray
        Meshgrid coordinates corresponding to the image dimensions.
    warp_matrix : ndarray
        Current transformation matrix.
    motion_type : str, optional
        Type of transformation: "translation", "affine", "euclidean", "homography".
        Default is "affine".

    Returns
    -------
    ndarray
        The computed Jacobian matrix, with shape depending on the motion model and image dimensionality.

    Notes
    -----
    - The returned Jacobian is used in ECC optimization to update transformation parameters.
    - For 2D images, the Jacobian shape is (num_params, H, W).
    - For 3D images, the Jacobian shape is (num_params, D, H, W).
    - The `grad` parameter should be a sequence (such as a list or array) containing the spatial gradients of the warped image.
      For 2D images, this is typically [grad_x, grad_y], and for 3D images, [grad_x, grad_y, grad_z]. Each gradient array
      should have the same shape as the warped image and represent the rate of change of pixel intensities along the respective axis.

    """
    def compute_jacobian_translation(grad):
        grad_iw_x, grad_iw_y = grad
        return np.stack([grad_iw_x, grad_iw_y])

    def compute_jacobian_translation_3D(grad):
        grad_iw_x, grad_iw_y, grad_iw_z = grad
        return np.stack([grad_iw_x, grad_iw_y, grad_iw_z])

    def compute_jacobian_affine(grad, xy_grid):
        grad_iw_x, grad_iw_y = grad
        x_grid, y_grid = xy_grid

        return np.stack(
            [
                grad_iw_x * x_grid,
                grad_iw_y * x_grid,
                grad_iw_x * y_grid,
                grad_iw_y * y_grid,
                grad_iw_x,
                grad_iw_y,
            ]
        )

    def compute_jacobian_affine_3D(grad, xy_grid):
        grad_iw_x, grad_iw_y, grad_iw_z = grad
        x_grid, y_grid, z_grid = xy_grid

        return np.stack(
            [
                grad_iw_x * x_grid,
                grad_iw_y * x_grid,
                grad_iw_z * x_grid,
                grad_iw_x * y_grid,
                grad_iw_y * y_grid,
                grad_iw_z * y_grid,
                grad_iw_x * z_grid,
                grad_iw_y * z_grid,
                grad_iw_z * z_grid,
                grad_iw_x,
                grad_iw_y,
                grad_iw_z,
            ]
        )

    def compute_jacobian_euclidean(grad, xy_grid, warp_matrix):
        grad_iw_x, grad_iw_y = grad
        x_grid, y_grid = xy_grid

        h0 = warp_matrix[0, 0]
        h1 = warp_matrix[0, 1]

        hat_x = -(x_grid * h1) - (y_grid * h0)
        hat_y = (x_grid * h0) - (y_grid * h1)

        return np.stack([grad_iw_x * hat_x + grad_iw_y * hat_y, grad_iw_x, grad_iw_y])

    def compute_jacobian_homography(grad, xy_grid, warp_matrix):
        # TODO: Lets look at the paper to see if this can be made cleaner using Numpy broadcasting
        h0_ = warp_matrix[0, 0]
        h1_ = warp_matrix[1, 0]
        h2_ = warp_matrix[2, 0]
        h3_ = warp_matrix[0, 1]
        h4_ = warp_matrix[1, 1]
        h5_ = warp_matrix[2, 1]
        h6_ = warp_matrix[0, 2]
        h7_ = warp_matrix[1, 2]

        grad_iw_x, grad_iw_y = grad
        x_grid, y_grid = xy_grid

        den_ = x_grid * h2_ + y_grid * h5_ + 1.0

        grad_iw_x_ = grad_iw_x / den_
        grad_iw_y_ = grad_iw_y / den_

        hat_x = -x_grid * h0_ - y_grid * h3_ - h6_
        hat_x = np.divide(hat_x, den_)

        hat_y = -x_grid * h1_ - y_grid * h4_ - h7_
        hat_y = np.divide(hat_y, den_)

        temp = hat_x * grad_iw_x_ + hat_y * grad_iw_y_

        return np.stack(
            [
                grad_iw_x_ * x_grid,
                grad_iw_y_ * x_grid,
                temp * x_grid,
                grad_iw_x_ * y_grid,
                grad_iw_y_ * y_grid,
                temp * y_grid,
                grad_iw_x_,
                grad_iw_y_,
            ]
        )

    if np.shape(grad)[0] == 2:
        match motion_type:
            case "translation":
                return compute_jacobian_translation(grad)
            case "affine":
                return compute_jacobian_affine(grad, xy_grid)
            case "euclidean":
                return compute_jacobian_euclidean(grad, xy_grid, warp_matrix)
            case "homography":
                return compute_jacobian_homography(grad, xy_grid, warp_matrix)
    else:
        match motion_type:
            case "translation":
                return compute_jacobian_translation_3D(grad)
            case "affine":
                return compute_jacobian_affine_3D(grad, xy_grid)


def update_warping_matrix(map_matrix, update, motion_type="affine"):
    """
    Updates the transformation matrix by applying the parameter increments computed during ECC optimization.

    This function supports multiple motion models, including translation, affine, euclidean, homography,
    and their 3D variants. The update is performed by adding the parameter increments to the appropriate
    elements of the transformation matrix, according to the selected motion type.

    Parameters
    ----------
    map_matrix : ndarray
        The current transformation matrix to be updated. For 2D images, this is typically a 3x3 matrix.
        For 3D images, this is typically a 4x4 matrix.
    update : ndarray or sequence
        The parameter increments to be applied to the transformation matrix. The length and meaning of
        this vector depend on the motion type.
    motion_type : str, optional
        The type of transformation model. Supported values are "translation", "affine", "euclidean",
        "homography", and their 3D variants. Default is "affine".

    Returns
    -------
    ndarray
        The updated transformation matrix.
    """

    def update_warping_matrix_translation(map_matrix, update):
        map_matrix[0, 2] += update[0]
        map_matrix[1, 2] += update[1]
        return map_matrix

    def update_warping_matrix_translation_3D(map_matrix, update):
        map_matrix[0, 3] += update[0]
        map_matrix[1, 3] += update[1]
        map_matrix[2, 3] += update[2]
        return map_matrix

    def update_warping_matrix_affine(map_matrix, update):
        map_matrix[0, 0] += update[0]
        map_matrix[1, 0] += update[1]
        map_matrix[0, 1] += update[2]
        map_matrix[1, 1] += update[3]
        map_matrix[0, 2] += update[4]
        map_matrix[1, 2] += update[5]
        return map_matrix

    def update_warping_matrix_euclidean(map_matrix, update):
        new_theta = update[0]
        new_theta += np.arcsin(map_matrix[1, 0])

        map_matrix[0, 2] += update[1]
        map_matrix[1, 2] += update[2]
        map_matrix[0, 0] = np.cos(new_theta)
        map_matrix[1, 1] = map_matrix[0, 0]
        map_matrix[1, 0] = np.sin(new_theta)
        map_matrix[0, 1] = -map_matrix[1, 0]
        return map_matrix

    def update_warping_matrix_homography(map_matrix, update):
        map_matrix[0, 0] += update[0]
        map_matrix[1, 0] += update[1]
        map_matrix[2, 0] += update[2]
        map_matrix[0, 1] += update[3]
        map_matrix[1, 1] += update[4]
        map_matrix[2, 1] += update[5]
        map_matrix[0, 2] += update[6]
        map_matrix[1, 2] += update[7]
        return map_matrix

    def update_warping_matrix_affine_3D(map_matrix, update):
        map_matrix[0, 0] += update[0]
        map_matrix[1, 0] += update[1]
        map_matrix[2, 0] += update[2]
        map_matrix[0, 1] += update[3]
        map_matrix[1, 1] += update[4]
        map_matrix[2, 1] += update[5]
        map_matrix[0, 2] += update[6]
        map_matrix[1, 2] += update[7]
        map_matrix[2, 2] += update[8]
        map_matrix[0, 3] += update[9]
        map_matrix[1, 3] += update[10]
        map_matrix[2, 3] += update[11]
        return map_matrix

    if np.shape(map_matrix)[0] == 3:
        match motion_type:
            case "translation":
                return update_warping_matrix_translation(map_matrix, update)
            case "affine":
                return update_warping_matrix_affine(map_matrix, update)
            case "euclidean":
                return update_warping_matrix_euclidean(map_matrix, update)
            case "homography":
                return update_warping_matrix_homography(map_matrix, update)
    else:
        match motion_type:
            case "translation":
                return update_warping_matrix_translation_3D(map_matrix, update)
            case "affine":
                return update_warping_matrix_affine_3D(map_matrix, update)


def project_onto_jacobian(jac, mat):
    """
    Projects a matrix onto the Jacobian by computing the sum of element-wise products.

    This function multiplies the Jacobian array and the input matrix element-wise and then sums over all spatial axes.
    The result is a vector where each element represents the projection of the input matrix onto one parameter direction
    of the Jacobian.

    Parameters
    ----------
    jac : ndarray
        The Jacobian matrix, typically of shape (num_params, H, W) for 2D images or (num_params, D, H, W) for 3D images.
    mat : ndarray
        The matrix to project, typically of shape (H, W) for 2D or (D, H, W) for 3D.

    Returns
    -------
    ndarray
        The projection result, a vector of length num_params.

    Notes
    -----
    - In the original code, the Jacobian was stored as a 2D [K*H, W] array, and the code looped through K, splitting the matrix into K submatrices.
        Each submatrix and the input matrix were flattened into vectors, and a dot product was applied.
        This is equivalent to multiplying the two matrices together element-by-element, then summing the result.
        Here, the Jacobian is stored as a 3D array [K, H, W], so broadcasting is used to avoid explicit looping.
        The summation is performed over all spatial axes, which generalizes to higher dimensions.
    """
    axis_summation = tuple(np.arange(1, len(np.shape(jac))))
    return np.sum(
        np.multiply(jac, mat), axis=axis_summation
    )  # axis=(1, 2)) if 2D, axis=(1, 2, 3)) if 3D


def compute_hessian(jac):
    """
    Computes the Hessian matrix from the Jacobian for ECC optimization.

    The Hessian matrix quantifies the second-order partial derivatives of the cost function with respect to the transformation parameters.
    It is constructed by summing the outer products of the Jacobian across all spatial axes, resulting in a symmetric matrix
    that is used to update the transformation parameters during optimization.

    Parameters
    ----------
    jac : ndarray
        The Jacobian matrix, typically of shape (num_params, H, W) for 2D images or (num_params, D, H, W) for 3D images.

    Returns
    -------
    ndarray
        The Hessian matrix, of shape (num_params, num_params).

    Notes
    -----
        The implementation uses `np.tensordot` to sum the products of the Jacobian over all spatial axes, which generalizes to higher dimensions.
    This is equivalent to the following code::

        hessian = np.empty((np.shape(jac)[0], np.shape(jac)[0]))
        for i in range(np.shape(jac)[0]):
            hessian[i,:] = np.sum(np.multiply(jac[i,:,:], jac), axis=(1,2))
            for j in range(i+1, np.shape(jac)[0]):
                hessian[i,j] = np.sum(np.multiply(jac[i,:,:], jac[j,:,:]))
                hessian[j,i] = hessian[i,j]
    
    """
    
    axis_summation = tuple(np.arange(1, len(np.shape(jac))))
    hessian = np.tensordot(
        jac, jac, axes=((axis_summation, axis_summation))
    )  # axes=([1, 2], [1, 2])) if 2D
    # axes=([1, 2, 3], [1, 2, 3]) if 3D
    return hessian