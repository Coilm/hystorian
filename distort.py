from math import asin, cos, sin
from typing import Literal

import numpy as np
from scipy.signal import convolve2d
from skimage import transform
from skimage.filters import gaussian as sk_gaussian
from skimage.transform import AffineTransform
from skimage.transform import warp as sk_warp

MOTIONS_TYPES = Literal["translation", "euclidean", "affine", "homography"]


def find_transform_ECC(
    src,
    dst,
    map=None,
    motion_type: MOTIONS_TYPES = "translation",
    criteria=(200, -1),
    input_mask=None,
    gauss_filt_size=None,
):
    if src.dtype != dst.dtype:
        raise TypeError("src and dst images must have the same dtype")

    if map is None:
        """
        row_count = 2
        if motion_type == "homography":
            row_count = 3
        """
        map = np.eye(3, dtype=np.float32)

    num_of_iterations = criteria[0]
    termination_eps = criteria[1]

    hs, ws = src.shape
    hd, wd = dst.shape

    Xcoord = np.array(range(ws))
    Ycoord = np.array(range(hs))
    Xgrid = []
    Ygrid = []

    for _ in range(hs):
        Xgrid.append(Xcoord)
    for _ in range(ws):
        Ygrid.append(Ycoord)

    Xgrid = np.array(Xgrid, dtype=np.float32)
    Ygrid = np.array(Ygrid, dtype=np.float32)

    dx = np.array([[-0.5, 0.0, 0.5]])

    gradient_x, gradient_y = np.gradient(dst, edge_order=2)
    # gradient_y = np.gradient(dst, axis=1)
    # gradient_x = convolve2d(dst, dx, mode="same")
    # gradient_y = convolve2d(dst, dx.transpose(), mode="same")

    rho = -1
    last_rho = -termination_eps

    for loop_count in range(num_of_iterations):
        print(loop_count)
        if abs(rho - last_rho) < termination_eps:
            break

        # Used because sk_warp is way faster when an AffineTransform is passed.
        if motion_type != "homography":
            affine_map = AffineTransform(map)
            gradientX_warped = sk_warp(gradient_x, affine_map)
            gradientY_warped = sk_warp(gradient_y, affine_map)
            image_warped = sk_warp(dst, affine_map)
        else:
            gradientX_warped = sk_warp(gradient_x, map)
            gradientY_warped = sk_warp(gradient_y, map)
            image_warped = sk_warp(dst, map)

        if input_mask is not None:
            if gauss_filt_size is not None:
                input_mask = sk_gaussian(input_mask, sigma=gauss_filt_size)

            input_mask = sk_warp(input_mask, map)

            image_warped = np.ma.array(image_warped, mask=input_mask)
            src = np.ma.array(src, mask=input_mask)

        image_warped = image_warped - np.mean(image_warped)
        template_zm = src - np.mean(src)

        img_norm = np.linalg.norm(image_warped)  # np.sqrt(img_count * img_std**2)
        src_norm = np.linalg.norm(src)  # np.sqrt(src_count * src_std**2)

        match motion_type:
            case "translation":
                jacobian = image_jacobian_translation_ECC(gradientX_warped, gradientY_warped)
            case "euclidean":
                jacobian = image_jacobian_euclidean_ECC(gradientX_warped, gradientY_warped, Xgrid, Ygrid, map)
            case "affine":
                jacobian = image_jacobian_affine_ECC(gradientX_warped, gradientY_warped, Xgrid, Ygrid)
            case "homography":
                jacobian = image_jacobian_homo_ECC(gradientX_warped, gradientY_warped, Xgrid, Ygrid, map)

        hessian = project_onto_jacobian_ECC(jacobian, jacobian)
        hessian_inv = np.linalg.inv(hessian)

        correlation = template_zm.ravel().dot(image_warped.ravel())

        last_rho = rho
        rho = correlation / (src_norm * img_norm)

        if np.isnan(rho):
            raise ValueError("NaN detected. Stopping ECC optimization.")

        image_projection = project_onto_jacobian_ECC(jacobian, image_warped)
        template_projection = project_onto_jacobian_ECC(jacobian, template_zm)

        image_projection_hessian = hessian_inv @ image_projection

        lambda_n = img_norm**2 - image_projection.ravel().dot(image_projection_hessian.ravel())

        lambda_d = correlation - template_projection.ravel().dot(image_projection_hessian.ravel())

        if lambda_d < 0:
            rho = -1
            raise ValueError(
                f"The algorithm stopped before its convergence. lambda_d negative: {lambda_d}\n The correlation is going to be minimized. Images may be uncorrelated or non-overlapped."
            )

        lambda_ = lambda_n / lambda_d

        error = lambda_ * template_zm - image_warped
        error_projection = project_onto_jacobian_ECC(jacobian, error)
        delta_p = hessian_inv * error_projection
        print(delta_p)
        map = update_warping_matrix_ECC(map, delta_p, motion_type)

    return map


def image_jacobian_translation_ECC(src1, src2):
    w = src1.shape[1]
    dst = np.empty((src1.shape[0], 2 * src1.shape[1]), dtype=np.float32)
    dst[:, 0:w] = src1
    dst[:, w : 2 * w] = src2

    return dst


def image_jacobian_euclidean_ECC(src1, src2, src3, src4, src5):
    w = src1.shape[1]

    dst = np.empty((src1.shape[0], 3 * src1.shape[1]), dtype=np.float32)

    _src5 = src5.ravel()
    h0 = _src5[0]
    h1 = _src5[3]

    hat_x = -src3 * h1 - src4 * h0
    hat_y = src3 * h0 - src4 * h1

    dst[:, 0:w] = src1[:, 0:w] = src1 @ hat_x + src2 @ hat_y

    dst[:, w : 2 * w] = src1
    dst[:, 2 * w : 3 * w] = src2

    return dst


def image_jacobian_affine_ECC(src1, src2, src3, src4):
    w = src1.shape[1]
    dst = np.empty((src1.shape[0], 6 * src1.shape[1]), dtype=np.float32)

    dst[:, 0:w] = src1 @ src3
    dst[:, w : 2 * w] = src2 @ src3
    dst[:, 2 * w : 3 * w] = src1 @ src4
    dst[:, 3 * w : 4 * w] = src2 @ src4
    dst[:, 4 * w : 5 * w] = src1
    dst[:, 5 * w : 6 * w] = src2

    return dst


def image_jacobian_homo_ECC(src1, src2, src3, src4, src5):
    w = src1.shape[1]
    dst = np.empty((src1.shape[0], 8 * src1.shape[1]), dtype=np.float32)

    _src5 = src5.ravel()

    h0 = _src5[0]
    h1 = _src5[3]
    h2 = _src5[6]
    h3 = _src5[1]
    h4 = _src5[4]
    h5 = _src5[7]
    h6 = _src5[2]
    h7 = _src5[5]

    den = src2 * h2 + src4 * h5 + 1.0

    hat_x = -src3 * h0 - src4 * h3 - h6
    hat_x /= den

    hat_y = -src3 * h1 - src4 * h4 - h7
    hat_y /= den

    src1_div = src1 / den
    src2_div = src2 / den

    tmp_ = hat_x @ src1_div + hat_y @ src2_div
    dst[:, 0:w] = src1_div @ src3
    dst[:, w : 2 * w] = src2_div @ src3
    dst[:, 2 * w : 3 * w] = tmp_ @ src3
    dst[:, 3 * w : 4 * w] = src1_div @ src4
    dst[:, 4 * w : 5 * w] = src2_div @ src4
    dst[:, 5 * w : 6 * w] = tmp_ @ src4
    dst[:, 6 * w : 7 * w] = src1_div
    dst[:, 7 * w : 8 * w] = src2_div

    return dst


def project_onto_jacobian_ECC(src1, src2):
    if src1.shape[1] != src2.shape[1]:
        w = src2.shape[1]
        dst = []
        src2_ = src2.ravel()
        for i in range(src1.shape[1] // src2.shape[1]):
            tmp = src2_.dot(src1[:, i * w : (i + 1) * w].ravel())
            dst.append(tmp)

        return np.array(dst)

    dst = np.empty((src1.shape[1] // src1.shape[0], src1.shape[1] // src1.shape[0]))
    w = src2.shape[1] // dst.shape[1]
    dst_ = np.ravel(dst)

    for i in range(dst.shape[1]):
        mat = src1[:, i * w : (i + 1) * w]
        dst_[i * (dst.shape[0] + 1)] = np.power(np.linalg.norm(mat), 2)

        for j in range(i + 1, dst.shape[1]):
            dst_[i * dst.shape[1] + j] = mat.ravel().dot(src2[:, j * w : (j + 1) * w].ravel())
            dst_[j * dst.shape[1] + i] = dst_[i * dst.shape[1] + j]

    dst = dst_.reshape((src1.shape[1] // src1.shape[0], src1.shape[1] // src1.shape[0]))
    return dst


def update_warping_matrix_ECC(map, update, motion_type: MOTIONS_TYPES):
    map_ = np.ravel(map)
    update_ = np.ravel(update)

    match motion_type:
        case "translation":
            map_[2] += update_[0]
            map_[5] += update_[1]
        case "affine":
            map_[0] += update_[0]
            map_[3] += update_[1]
            map_[1] += update_[2]
            map_[4] += update_[3]
            map_[2] += update_[4]
            map_[5] += update_[5]
        case "homography":
            map_[0] += map_[0]
            map_[3] += map_[1]
            map_[6] += map_[2]
            map_[1] += map_[3]
            map_[4] += map_[4]
            map_[7] += map_[5]
            map_[2] += map_[6]
            map_[5] += map_[7]
        case "euclidean":
            new_theta = map_[0]
            new_theta += asin(map_[3])

            map_[2] += map_[1]
            map_[5] += map_[2]
            map_[0] = cos(new_theta)
            map_[4] = cos(new_theta)
            map_[3] = sin(new_theta)
            map_[1] = -map_[3]

    return map_.reshape(map.shape)
