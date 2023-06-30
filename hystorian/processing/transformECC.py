import math
from enum import Enum

from scipy.signal import convolve2d
from skimage.filters import gaussian as sk_gaussian
from skimage.transform import AffineTransform
from skimage.transform import warp as sk_warp


class MotionType(Enum):
    MOTION_TRANSLATION = "TRANSLATION"
    MOTION_EUCLIDEAN = "EUCLIDEAN"
    MOTION_AFFINE = "AFFINE"
    MOTION_HOMOGRAPHY = "HOMOGRAPHY"


class CriteriaType(Enum):
    COUNT = 1
    MAX_ITER = COUNT
    EPS = 2


class TermCriteria:
    def __init__(self, _type=CriteriaType.COUNT, maxCount=0, epsilon=0.0):
        self.type = _type
        self.maxCount = maxCount
        self.epsilon = epsilon

    def isValid(self):
        isCount = (self.type and CriteriaType.COUNT) and self.maxCount > 0
        isEps = (self.type and CriteriaType.EPS) and not math.isnan(self.epsilon)
        return isCount or isEps

    def __str__(self):
        return "TermCriteria: type: %d, maxCount: %d, epsilon: %f" % (self.type, self.maxCount, self.epsilon)


def find_transform_ECC(template_image, input_image, warp_matrix, motion_type, criteria, input_mask, gauss_filt_size):
    src = template_image
    dst = input_image
    map = warp_matrix

    if np.any(map):  # if map is not empty
        row_count = 2
        if motion_type == MotionType.MOTION_HOMOGRAPHY:
            row_count = 3

        map = np.eye(row_count, 3, dtype=np.float32)

    if src.dtype != dst.dtype:
        raise TypeError("src and dst must have the same dtype")

    numberOfIterations = criteria.maxCount if criteria.type & CriteriaType.COUNT else 200
    termination_eps = criteria.epsilon if criteria.type & CriteriaType.EPS else -1

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

    if np.any(input_mask):
        pre_mask = np.ones((hd, wd), dtype=np.uint8)
    else:
        pre_mask = np.zeros(input_mask.shape, dtype=np.uint8)
        pre_mask[input_mask > 0] = 1

    pre_mask = sk_gaussian(pre_mask.astype(np.float32), sigma=gauss_filt_size)

    gradientX = np.zeros((hd, wd), dtype=np.float32)
    gradientY = np.zeros((hd, wd), dtype=np.float32)
    dx = np.array([-0.5, 0.0, 0.5])

    image_float = dst.astype(np.float32)

    gradientX = convolve2d(image_float, dx)
    gradientY = convolve2d(image_float, dx.transpose())

    rho = -1
    last_rho = -termination_eps

    for i in range(numberOfIterations):
        if abs(rho - last_rho) < termination_eps:
            break

        if motion_type != MotionType.MOTION_HOMOGRAPHY:
            map = AffineTransform(map)

        image_warped = sk_warp(image_float, map)
        gradientX_warped = sk_warp(gradientX, map)
        gradientY_warped = sk_warp(gradientY, map)
        image_mask = sk_warp(pre_mask, map)

        image_warped = np.ma.array(image_warped, mask=image_mask)
        template_image = np.ma.array(template_image.astype(float), mask=image_mask)

        img_mean = np.ma.mean(image_warped)
        img_std = np.ma.std(image_warped)
        tmp_mean = np.ma.mean(template_image)
        tmp_std = np.ma.std(template_image)

        image_warped = image_warped - img_mean
        template_zm = template_image - tmp_mean

        img_norm = np.ma.sqrt(np.sum(image_mask != 0) * img_std**2)
        tmp_norm = np.ma.sqrt(np.sum(image_mask != 0) * tmp_std**2)

        if motion_type == MotionType.MOTION_AFFINE:
            jacobian = image_jacobian_affine_ECC(gradientX_warped, gradientY_warped, Xgrid, Ygrid)
        elif motion_type == MotionType.MOTION_HOMOGRAPHY:
            jacobian = image_jacobian_homo_ECC(gradientX_warped, gradientY_warped, Xgrid, Ygrid, map)
        elif motion_type == MotionType.MOTION_TRANSLATION:
            jacobian = image_jacobian_translation_ECC(gradientX_warped, gradientY_warped)
        elif motion_type == MotionType.MOTION_EUCLIDEAN:
            jacobian = image_jacobian_euclidean_ECC(gradientX_warped, gradientY_warped, Xgrid, Ygrid, map)
        else:
            raise ValueError("motion_type is not a valid value: {motion_type}.")

        hessian = project_onto_jacobian_ECC(jacobian, jacobian)
        hessian_inv = np.linalg.inv(hessian)

        correlation = np.inner(template_zm, image_warped)

        last_rho = rho
        rho = correlation / (img_norm * tmp_norm)

        if np.isnan(rho):
            raise ValueError("Nan encountered.")

        image_projection = project_onto_jacobian_ECC(jacobian, image_warped)
        template_projection = project_onto_jacobian_ECC(jacobian, template_zm)

        image_projection_hessian = hessian_inv * image_projection
        lambda_n = np.inner(img_norm, img_norm) - np.inner(image_projection, image_projection_hessian)
        lambda_d = correlation - np.inner(template_projection, image_projection_hessian)

        if lambda_d < 0:
            rho = -1
            raise ValueError(
                "The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped."
            )

        lambda_ = lambda_n / lambda_d

        error = lambda_ * template_zm - image_warped
        error_projection = project_onto_jacobian_ECC(jacobian, error)
        delta_p = hessian_inv * error_projection

        update_warping_matrix_ECC(map, delta_p, motion_type)


def project_onto_jacobian_ECC(src1, src2):
    if src1.shape[1] != src2.shape[1]:
        # dst.cols == 1
        w = src2.shape[1]
        dst = np.zeros((src1.shape[1], 1), dtype=np.float32)
        for i in range(dst.shape[0]):
            dst[i, 0] = np.dot(src2, src1[:, i * w : (i + 1) * w])
    else:
        # dst is square (and symmetric)
        dst = np.zeros((src1.shape[1], src1.shape[1]), dtype=np.float32)
        w = src2.shape[1] // dst.shape[1]
        for i in range(dst.shape[0]):
            mat = src1[:, i * w : (i + 1) * w]
            dst[i, i] = np.linalg.norm(mat) ** 2  # diagonal elements
            for j in range(i + 1, dst.shape[1]):
                dst[i, j] = np.dot(mat, src2[:, j * w : (j + 1) * w])
                dst[j, i] = dst[i, j]  # due to symmetry
    return dst


def image_jacobian_affine_ECC(src1, src2, src3, src4):
    w = src1.shape[1]
    dst = np.zeros((src1.shape[0], 6 * w), dtype=np.float32)

    # Compute Jacobian blocks (6 blocks)
    dst[:, 0:w] = src1 * src3  # 1
    dst[:, w : 2 * w] = src2 * src3  # 2
    dst[:, 2 * w : 3 * w] = src1 * src4  # 3
    dst[:, 3 * w : 4 * w] = src2 * src4  # 4
    dst[:, 4 * w : 5 * w] = src1  # 5
    dst[:, 5 * w : 6 * w] = src2  # 6

    return dst


def image_jacobian_homo_ECC(src1, src2, src3, src4, src5):
    hptr = src5.ravel()  # Flatten the array

    h0_ = hptr[0]
    h1_ = hptr[3]
    h2_ = hptr[6]
    h3_ = hptr[1]
    h4_ = hptr[4]
    h5_ = hptr[7]
    h6_ = hptr[2]
    h7_ = hptr[5]

    w = src1.shape[1]

    # Create denominator for all points as a block
    den_ = src3 * h2_ + src4 * h5_ + 1.0

    # Create projected points
    hatX_ = -src3 * h0_ - src4 * h3_ - h6_
    hatX_ /= den_
    hatY_ = -src3 * h1_ - src4 * h4_ - h7_
    hatY_ /= den_

    # Instead of dividing each block with den_, pre-divide the block of gradients
    src1Divided_ = np.divide(src1, den_)
    src2Divided_ = np.divide(src2, den_)

    # Compute Jacobian blocks (8 blocks)
    dst = np.zeros((src1.shape[0], src1.shape[1] * 8), dtype=np.float32)
    dst[:, 0:w] = src1Divided_ * src3  # 1
    dst[:, w : 2 * w] = src2Divided_ * src3  # 2
    temp_ = (hatX_ * src1Divided_) + (hatY_ * src2Divided_)
    dst[:, 2 * w : 3 * w] = temp_ * src3  # 3
    dst[:, 3 * w : 4 * w] = src1Divided_ * src4  # 4
    dst[:, 4 * w : 5 * w] = src2Divided_ * src4  # 5
    dst[:, 5 * w : 6 * w] = temp_ * src4  # 6
    dst[:, 6 * w : 7 * w] = src1Divided_  # 7
    dst[:, 7 * w : 8 * w] = src2Divided_  # 8

    return dst


def image_jacobian_translation_ECC(src1, src2):
    w = src1.shape[1]

    # Compute Jacobian blocks (2 blocks)
    dst = np.zeros((src1.shape[0], w * 2), dtype=src1.dtype)
    dst[:, 0:w] = src1
    dst[:, w : 2 * w] = src2

    return dst


import numpy as np


def image_jacobian_euclidean_ECC(src1, src2, src3, src4, src5):
    w = src1.shape[1]

    # Create -sin(theta)*X -cos(theta)*Y for all points as a block -> hatX
    hptr = src5.ravel()
    h0 = hptr[0]  # cos(theta)
    h1 = hptr[3]  # sin(theta)
    hatX = -(src3 * h1) - (src4 * h0)

    # Create cos(theta)*X -sin(theta)*Y for all points as a block -> hatY
    hatY = (src3 * h0) - (src4 * h1)

    # Compute Jacobian blocks (3 blocks)
    dst = np.zeros((src1.shape[0], w * 3), dtype=src1.dtype)
    dst[:, 0:w] = (src1 * hatX) + (src2 * hatY)
    dst[:, w : 2 * w] = src1
    dst[:, 2 * w : 3 * w] = src2

    return dst


def update_warping_matrix_ECC(map_matrix, update, motionType):
    map_matrix = np.copy(map_matrix)
    update = np.copy(update)

    if motionType == MotionType.MOTION_TRANSLATION:
        map_matrix[1, 2] += update[1]
        map_matrix[0, 2] += update[0]
    elif motionType == MotionType.MOTION_AFFINE:
        map_matrix[0, 0] += update[0]
        map_matrix[0, 1] += update[2]
        map_matrix[0, 2] += update[4]
        map_matrix[1, 0] += update[1]
        map_matrix[1, 1] += update[3]
        map_matrix[1, 2] += update[5]
    elif motionType == MotionType.MOTION_HOMOGRAPHY:
        map_matrix[0, 0] += update[0]
        map_matrix[0, 1] += update[3]
        map_matrix[0, 2] += update[6]
        map_matrix[1, 0] += update[1]
        map_matrix[1, 1] += update[4]
        map_matrix[1, 2] += update[7]
        map_matrix[2, 0] += update[2]
        map_matrix[2, 1] += update[5]
    elif motionType == MotionType.MOTION_EUCLIDEAN:
        new_theta = update[0] + math.asin(map_matrix[1, 0])
        map_matrix[0, 0] = map_matrix[1, 1] = np.cos(new_theta)
        map_matrix[1, 0] = -np.sin(new_theta)
        map_matrix[0, 2] += update[1]
        map_matrix[1, 2] += update[2]
    return map_matrix
