from math import asin, cos, sin
from typing import Literal

import numpy as np
from scipy.signal import convolve2d
from skimage import transform
from skimage.filters import gaussian as sk_gaussian
from skimage.transform import AffineTransform
from skimage.transform import warp as sk_warp

MOTIONS_TYPES = Literal["translation", "euclidean", "affine", "homography"]


def generate_warp_matrix(img=None, offset=[0, 0], angle=0, scale=[1, 1], shear=[0, 0]):
    C = np.eye(3, 3, dtype=np.float32)
    Cinv = np.eye(3, 3, dtype=np.float32)

    if img is not None:
        C[0, 2] = -0.5 * img.shape[1]
        C[1, 2] = -0.5 * img.shape[0]

        Cinv[0, 2] = 0.5 * img.shape[1]
        Cinv[1, 2] = 0.5 * img.shape[0]

    T = np.eye(3, 3, dtype=np.float32)
    T[0, 2] = offset[0]
    T[1, 2] = offset[1]

    S = np.eye(3, 3, dtype=np.float32)
    S[0, 0] = scale[0]
    S[1, 1] = scale[1]

    Z = np.eye(3, 3, dtype=np.float32)
    Z[0, 1] = shear[0]
    Z[1, 0] = shear[1]

    R = np.eye(3, 3, dtype=np.float32)
    R[0, 0] = np.cos(np.deg2rad(angle))
    R[0, 1] = -np.sin(np.deg2rad(angle))
    R[1, 0] = np.sin(np.deg2rad(angle))
    R[1, 1] = np.cos(np.deg2rad(angle))

    warp_matrix = np.matmul(T, np.matmul(np.matmul(Cinv, np.matmul(R, np.matmul(S, Z))), C))

    return warp_matrix


def warp_image(
    img,
    tform,
    order=None,
    mode: Literal["constant", "edge", "symmetric", "reflect", "wrap"] = "constant",
    cval=0.0,
    preserve_range=False,
):
    tmp2 = np.copy(tform)

    if tform.shape[0] != tform.shape[1]:
        tmp2 = np.eye(3)
        tmp2[:2, :] = tform

    out_img = transform.warp(img, tmp2, mode=mode, cval=cval, preserve_range=preserve_range)
    return out_img


def generate_transform_xy_single(
    img,
    img_orig,
    warp_matrix,
    warp_mode: MOTIONS_TYPES = "translation",
    termination_eps=1e-5,
    number_of_iterations=10000,
):
    criteria = (number_of_iterations, termination_eps)

    _warp_matrix = np.copy(warp_matrix)
    # _warp_matrix = _warp_matrix[:2, :]

    _warp_matrix = find_transform_ECC(img_orig, img, _warp_matrix, warp_mode, criteria, None, 1)

    return warp_matrix


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

    gradient_x = convolve2d(dst, dx, mode="same")
    gradient_y = convolve2d(dst, dx.transpose(), mode="same")

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

        img_mean = np.mean(image_warped)
        img_std = np.std(image_warped)
        src_mean = np.mean(src)
        src_std = np.std(src)

        image_warped = image_warped - img_mean
        template_zm = src - src_mean

        if np.ma.is_masked(image_warped):
            img_count = image_warped.count()
        else:
            img_count = image_warped.size

        if np.ma.is_masked(src):
            src_count = src.count()
        else:
            src_count = src.size

        img_norm = np.sqrt(img_count * img_std**2)
        src_norm = np.sqrt(src_count * src_std**2)

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

        correlation = np.trace(template_zm.transpose() @ image_warped)  # Frobenius inner product

        last_rho = rho
        rho = correlation / (src_norm * img_norm)

        if np.isnan(rho):
            raise ValueError("NaN detected. Stopping ECC optimization.")

        image_projection = project_onto_jacobian_ECC(jacobian, image_warped)
        template_projection = project_onto_jacobian_ECC(jacobian, template_zm)

        image_projection_hessian = hessian_inv * image_projection

        lambda_n = img_norm**2 - np.sum(image_projection.dot(image_projection_hessian))
        lambda_d = correlation - np.sum(template_projection.dot(image_projection_hessian))

        if lambda_d < 0:
            rho = -1
            raise ValueError(
                "The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped."
            )

        lambda_ = lambda_n / lambda_d

        error = lambda_ * template_zm - image_warped
        error_projection = project_onto_jacobian_ECC(jacobian, error)
        delta_p = hessian_inv * error_projection

        map = update_warping_matrix_ECC(map, delta_p, motion_type)
        print(map)
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
        for i in range(src1.shape[1] // src2.shape[1]):
            dst.append(np.trace(src2.transpose() @ src1[:, i * w : (i + 1) * w]))  # Frobenius inner product

        return np.array(dst)

    dst = np.empty((src1.shape[1] // src1.shape[0], src1.shape[1] // src1.shape[0]))
    w = src2.shape[1] // dst.shape[1]
    dst_ = np.ravel(dst)

    for i in range(dst.shape[1]):
        mat = src1[:, i * w : (i + 1) * w]
        dst_[i * (dst.shape[0] + 1)] = np.power(np.linalg.norm(mat), 2)

        for j in range(i + 1, dst.shape[1]):
            dst_[i * dst.shape[1] + j] = np.trace(
                mat.transpose() @ src2[:, j * w : (j + 1) * w]
            )  # Frobenius inner product
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
