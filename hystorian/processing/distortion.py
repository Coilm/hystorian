import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import warp, ProjectiveTransform


def generate_warp_matrix(img=None, offset=(0, 0), angle=0, scale=(1, 1), shear=(0, 0)):
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


def warp_image(img, tform, mode="constant", cval=0.0, preserve_range=False):
    tmp2 = np.copy(tform)

    if tform.shape[0] != tform.shape[1]:
        tmp2 = np.eye(3)
        tmp2[:2, :] = tform

    out_img = warp(img, tmp2, mode=mode, cval=cval, preserve_range=preserve_range)
    return out_img


def image_jacobian_homo_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, warpMatrix, jacobian):
    if gradientXWarped.size != gradientYWarped.size:
        raise Exception("gradientXWarped.size != gradientYWarped.size")
    if gradientXWarped.size != Xgrid.size:
        raise Exception("gradientXWarped.size != Xgrid.size")
    if gradientXWarped.size != Ygrid.size:
        raise Exception("gradientXWarped.size != Ygrid.size")

    if gradientXWarped.shape[0] != jacobian.shape[0]:
        raise Exception("gradientXWarped.shape[0] != jacobian.shape[0]")
    if jacobian.shape[1] != 8 * gradientXWarped.shape[1]:
        raise Exception("jacobian.shape[1] != 8*gradientXWarped.shape[1]")
    if jacobian.dtype != np.float32:
        raise Exception("jacobian.dtype != np.float32")

    # Not implemented because useless
    # CV_Assert(warpMatrix.isContinuous());

    h0_ = warpMatrix[0, 0]
    h1_ = warpMatrix[1, 0]
    h2_ = warpMatrix[2, 0]
    h3_ = warpMatrix[0, 1]
    h4_ = warpMatrix[1, 1]
    h5_ = warpMatrix[2, 1]
    h6_ = warpMatrix[0, 2]
    h7_ = warpMatrix[1, 2]

    w = gradientXWarped.shape[1]

    # create denominator for all points as a block
    den_ = Xgrid * h2_ + Ygrid * h5_ + 1.0  # check the time of this! otherwise use addWeighted

    # create projected points
    hatX_ = -Xgrid * h0_ - Ygrid * h3_ - h6_
    hatX_ = np.divide(hatX_, den_)

    hatY_ = -Xgrid * h1_ - Ygrid * h4_ - h7_
    hatY_ = np.divide(hatY_, den_)

    # instead of dividing each block with den,
    # just pre-divide the block of gradients (it's more efficient)

    gradientXWarpedDivided_ = np.divide(gradientXWarped, den_)
    gradientYWarpedDivided_ = np.divide(gradientYWarped, den_)

    temp_ = np.multiply(hatX_, gradientXWarpedDivided_) + np.multiply(hatY_, gradientYWarpedDivided_)

    # compute Jacobian blocks (8 blocks)
    jacobian[:, 0:w] = np.multiply(gradientXWarpedDivided_, Xgrid)  # 1
    jacobian[:, w : 2 * w] = np.multiply(gradientYWarpedDivided_, Xgrid)  # 2
    jacobian[:, 2 * w : 3 * w] = np.multiply(temp_, Xgrid)  # 3
    jacobian[:, 3 * w : 4 * w] = np.multiply(gradientXWarpedDivided_, Ygrid)  # 4
    jacobian[:, 4 * w : 5 * w] = np.multiply(gradientYWarpedDivided_, Ygrid)  # 5
    jacobian[:, 5 * w : 6 * w] = np.multiply(temp_, Ygrid)  # 6
    jacobian[:, 6 * w : 7 * w] = gradientXWarpedDivided_  # 7
    jacobian[:, 7 * w : 8 * w] = gradientYWarpedDivided_  # 8

    return jacobian


def image_jacobian_euclidean_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, warpMatrix, jacobian):
    if gradientXWarped.size != gradientYWarped.size:
        raise Exception("gradientXWarped.size != gradientYWarped.size")
    if gradientXWarped.size != Xgrid.size:
        raise Exception("gradientXWarped.size != Xgrid.size")
    if gradientXWarped.size != Ygrid.size:
        raise Exception("gradientXWarped.size != Ygrid.size")

    if gradientXWarped.shape[0] != jacobian.shape[0]:
        raise Exception("gradientXWarped.shape[0] != jacobian.shape[0]")
    if jacobian.shape[1] != 3 * gradientXWarped.shape[1]:
        raise Exception("jacobian.shape[1] != 3*gradientXWarped.shape[1]")
    if jacobian.dtype != np.float32:
        raise Exception("jacobian.dtype != np.float32")

    # Not implemented because useless
    # CV_Assert(warpMatrix.isContinuous());

    w = gradientXWarped.shape[1]

    h0 = warpMatrix[0, 0]  # cos(theta)
    h1 = warpMatrix[1, 0]  # sin(theta)

    # create -sin(theta)*X -cos(theta)*Y for all points as a block -> hatX
    hatX = -(Xgrid * h1) - (Ygrid * h0)

    # create cos(theta)*X -sin(theta)*Y for all points as a block -> hatY
    hatY = (Xgrid * h0) - (Ygrid * h1)

    # compute Jacobian blocks (3 blocks)
    jacobian[:, 0:w] = np.multiply(gradientXWarped, hatX) + np.multiply(gradientYWarped, hatY)  # 1
    jacobian[:, w : 2 * w] = gradientXWarped  # 2
    jacobian[:, 2 * w : 3 * w] = gradientYWarped  # 3

    return jacobian


def image_jacobian_affine_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, jacobian):
    if gradientXWarped.size != gradientYWarped.size:
        raise Exception("gradientXWarped.size != gradientYWarped.size")
    if gradientXWarped.size != Xgrid.size:
        raise Exception("gradientXWarped.size != Xgrid.size")
    if gradientXWarped.size != Ygrid.size:
        raise Exception("gradientXWarped.size != Ygrid.size")

    if gradientXWarped.shape[0] != jacobian.shape[0]:
        raise Exception("gradientXWarped.shape[0] != jacobian.shape[0]")
    if jacobian.shape[1] != 6 * gradientXWarped.shape[1]:
        raise Exception("jacobian.shape[1] != 6*gradientXWarped.shape[1]")

    if jacobian.dtype != np.float32:
        raise Exception("jacobian.dtype != np.float32")

    w = gradientXWarped.shape[1]

    # compute Jacobian blocks (6 blocks)

    jacobian[:, 0:w] = np.multiply(gradientXWarped, Xgrid)  # 1
    jacobian[:, w : 2 * w] = np.multiply(gradientYWarped, Xgrid)  # 2
    jacobian[:, 2 * w : 3 * w] = np.multiply(gradientXWarped, Ygrid)  # 3
    jacobian[:, 3 * w : 4 * w] = np.multiply(gradientYWarped, Ygrid)  # 4
    jacobian[:, 4 * w : 5 * w] = np.copy(gradientXWarped)  # 5
    jacobian[:, 5 * w : 6 * w] = np.copy(gradientYWarped)  # 6

    return jacobian


def image_jacobian_translation_ECC(gradientXWarped, gradientYWarped, jacobian):
    if gradientXWarped.size != gradientYWarped.size:
        raise Exception("gradientXWarped.size != gradientYWarped.size")

    if gradientXWarped.shape[0] != jacobian.shape[0]:
        raise Exception("gradientXWarped.shape[0] != jacobian.shape[0]")
    if jacobian.shape[1] != 2 * gradientXWarped.shape[1]:
        raise Exception("jacobian.shape[1] != 2*gradientXWarped.shape[1]")

    if jacobian.dtype != np.float32:
        raise Exception("jacobian.dtype != np.float32")

    w = gradientXWarped.shape[1]

    # compute Jacobian blocks (2 blocks)

    jacobian[:, 0:w] = gradientXWarped  # 1
    jacobian[:, w : 2 * w] = gradientYWarped  # 2

    return jacobian


def project_onto_jacobian_ECC(src1, src2, dst):

    # this functions is used for two types of projections. If src1.cols ==src.cols
    # it does a blockwise multiplication (like in the outer product of vectors)
    # of the blocks in matrices src1 and src2 and dst
    # has size (number_of_blcks x number_of_blocks), otherwise dst is a vector of size
    # (number_of_blocks x 1) since src2 is "multiplied"(dot) with each block of src1.

    # The number_of_blocks is equal to the number of parameters we are lloking for
    # (i.e. rtanslation:2, euclidean: 3, affine: 6, homography: 8)

    if src1.shape[0] != src2.shape[0]:
        raise Exception("src1.size != src2.size")
    if src1.shape[1] % src2.shape[1] != 0:
        raise Exception("src1.shape[1] % src2.shape[1] != 0")

    if src1.shape[1] != src2.shape[1]:  # dst.cols = 1
        w = src2.shape[1]
        for i in range(dst.shape[1]):
            dst[0, i] = np.vdot(src2, src1[:, i * w : (i + 1) * w])
    else:
        if dst.shape[0] != dst.shape[1]:
            raise Exception("dst.shape[0] != dst.shape[1]")
        w = int(src2.shape[1] / dst.shape[1])
        for i in range(dst.shape[1]):
            mat = src1[:, i * w : ((i + 1) * w)]
            dst[i, i] = np.linalg.norm(mat) ** 2

            for j in range(i + 1, dst.shape[1]):
                dst[j, i] = np.vdot(mat, src2[:, j * w : (j + 1) * w])
                dst[i, j] = dst[j, i]
    return dst


def update_warping_matrix_ECC(map_matrix, update, motionType):
    if map_matrix.dtype != np.float32:
        raise Exception("map_matrix.dtype != np.float32")
    if update.dtype != np.float32:
        raise Exception("update.dtype != np.float32")

    if motionType not in ["MOTION_TRANSLATION", "MOTION_EUCLIDEAN", "MOTION_AFFINE", "MOTION_HOMOGRAPHY"]:
        raise Exception("motionType not in ['MOTION_TRANSLATION','MOTION_EUCLIDEAN','MOTION_AFFINE','MOTION_HOMOGRAPHY']")

    if motionType == "MOTION_HOMOGRAPHY":
        if (map_matrix.shape[1] != 3) and (update.size != 8):
            raise Exception("(map_matrix.shape[1] != 3) and (update.size != 8)")
    elif motionType == "MOTION_AFFINE":
        if (map_matrix.shape[1] != 2) and (update.size != 6):
            raise Exception("(map_matrix.shape[1] != 2) and (update.size != 6)")
    elif motionType == "MOTION_EUCLIDEAN":
        if (map_matrix.shape[1] != 2) and (update.size != 3):
            raise Exception("(map_matrix.shape[1] != 2) and (update.size != 3)")
    else:
        if (map_matrix.shape[1] != 2) and (update.size != 2):
            raise Exception("(map_matrix.shape[1] != 2) and (update.size != 2)")

    if len(update.shape) != 1:
        raise Exception("len(update.shape) != 1")

    # Not implemented because useless
    # CV_Assert( map_matrix.isContinuous());
    # CV_Assert( update.isContinuous() );

    if motionType == "MOTION_TRANSLATION":
        map_matrix[0, 2] += update[0]
        map_matrix[1, 2] += update[1]

    if motionType == "MOTION_AFFINE":
        map_matrix[0, 0] += update[0]
        map_matrix[1, 0] += update[1]
        map_matrix[0, 1] += update[2]
        map_matrix[1, 1] += update[3]
        map_matrix[0, 2] += update[4]
        map_matrix[1, 2] += update[5]

    if motionType == "MOTION_HOMOGRAPHY":
        map_matrix[0, 0] += update[0]
        map_matrix[1, 0] += update[1]
        map_matrix[2, 0] += update[2]
        map_matrix[0, 1] += update[3]
        map_matrix[1, 1] += update[4]
        map_matrix[2, 1] += update[5]
        map_matrix[0, 2] += update[6]
        map_matrix[1, 2] += update[7]

    if motionType == "MOTION_EUCLIDEAN":
        new_theta = update[0]
        new_theta += np.arcsin(map_matrix[1, 0])

        map_matrix[0, 2] += update[1]
        map_matrix[1, 2] += update[2]
        map_matrix[0, 0] = map_matrix[1, 1] = np.cos(new_theta)
        map_matrix[1, 0] = np.sin(new_theta)
        map_matrix[0, 1] = -map_matrix[1, 0]
    return map_matrix


def computeECC(templateImage, inputImage, inputMask):
    if templateImage is None or templateImage.size == 0:
        raise Exception("templateImage is None or templateImage.size == 0")
    if inputImage is None or inputImage.size == 0:
        raise Exception("inputImage is None or inputImage.size == 0")

    if templateImage.dtype != inputImage.dtype:
        raise Exception("templateImage.type != inputImage.type")

    active_pixels = 0
    if inputMask is None:
        active_pixels = templateImage.shape[0] * templateImage.shape[1]
    else:
        active_pixels = np.sum(inputMask != 0)
    meanTemplate = np.mean(templateImage[inputMask != 0])
    sdTemplate = np.std(templateImage[inputMask != 0])
    templateImage_zeromean = np.zeros_like(templateImage)
    templateMat = templateImage
    inputMat = inputImage

    # For unsigned ints, when the mean is computed and subtracted, any values less than the mean
    # will be set to 0 (since there are no negatives values). This impacts the norm and dot product, which
    # ultimately results in an incorrect ECC. To circumvent this problem, if unsigned ints are provided,
    # we convert them to a signed ints with larger resolution for the subtraction step.

    if templateImage.dtype in [np.uint8, np.float32]:
        newtype = np.float32
        if templateImage.dtype == np.uint8:
            newtype = np.float32
        templateMat = templateImage.astype(newtype)
        inputMat = inputImage.astype(newtype)

    templateImage_zeromean[inputMask != 0] = templateMat[inputMask != 0] - meanTemplate

    templateImagenorm = np.sqrt(active_pixels * (sdTemplate**2))

    inputImage_zeromean = np.zeros_like(inputImage)
    meanInput = np.mean(inputImage[inputMask != 0])
    sdInput = np.std(inputImage[inputMask != 0])
    inputImage_zeromean[inputMask != 0] = inputMat[inputMask != 0] - meanInput
    inputImagenorm = np.sqrt(active_pixels * (sdInput**2))

    correlation = np.vdot(templateImage_zeromean, inputImage_zeromean)

    return correlation / (templateImagenorm * inputImagenorm)


def findTransformECC(
    templateImage,
    inputImage,
    warpMatrix=None,
    motionType="MOTION_AFFINE",
    numberOfIterations=200,
    termination_eps=-1,
    inputMask=None,
    gaussFiltSize=5,
):
    src = templateImage  # template image
    dst = inputImage  # input image (to be warped)

    if warpMatrix is None:
        warpMatrix = np.eye(3, dtype=np.float32)
        if motionType != "MOTION_HOMOGRAPHY":
            warpMatrix = warpMatrix[:2, :]

    if templateImage.dtype != inputImage.dtype:
        raise Exception("Both input images must have the same data type")

    if templateImage.dtype != np.uint8 and templateImage.dtype != np.float32:
        raise Exception("Images must have np.uint8 or np.float32 type")

    if warpMatrix.dtype != np.float32:
        raise Exception("warpMatrix must be single-channel floating-point matrix")

    if warpMatrix.shape[1] != 3:
        raise Exception("warpMatrix.shape[1] != 3")

    if warpMatrix.shape[0] != 3 and warpMatrix.shape[0] != 2:
        raise Exception("warpMatrix.shape[0] != 3 or warpMatrix.shape[0] != 2")

    if motionType not in ["MOTION_TRANSLATION", "MOTION_EUCLIDEAN", "MOTION_AFFINE", "MOTION_HOMOGRAPHY"]:
        raise Exception("motionType not in ['MOTION_TRANSLATION','MOTION_EUCLIDEAN','MOTION_AFFINE','MOTION_HOMOGRAPHY']")

    if motionType == "MOTION_HOMOGRAPHY" and warpMatrix.shape[0] != 3:
        raise Exception("motionType == 'MOTION_HOMOGRAPHY' and warpMatrix.shape[0] != 3")

    paramTemp = 6  # default: affine

    if motionType == "MOTION_TRANSLATION":
        paramTemp = 2
    elif motionType == "MOTION_EUCLIDEAN":
        paramTemp = 3
    elif motionType == "MOTION_HOMOGRAPHY":
        paramTemp = 8

    numberOfParameters = paramTemp

    ws = src.shape[1]
    hs = src.shape[0]
    wd = dst.shape[1]
    hd = dst.shape[0]

    Xcoord = np.arange(ws, dtype=np.float32)
    Ycoord = np.arange(hs, dtype=np.float32)

    Xgrid, Ygrid = np.meshgrid(Xcoord, Ycoord)
    Xgrid = Xgrid.astype(dtype=np.float32)
    Ygrid = Ygrid.astype(dtype=np.float32)

    templateZM = np.ndarray((hs, ws), dtype=np.float32)
    templateFloat = np.ndarray((hs, ws), dtype=np.float32)
    imageFloat = np.ndarray((hd, wd), dtype=np.float32)
    imageWarped = np.ndarray((hs, ws), dtype=np.float32)
    imageMask = np.ndarray((hs, ws), dtype=np.uint8)

    # to use it for mask warping
    preMask = np.ones((hd, wd), dtype=np.uint8)
    if inputMask is not None:
        preMask = inputMask

    # gaussian filtering is optional
    templateFloat = src.astype(templateFloat.dtype)
    templateFloat = gaussian_filter(templateFloat, gaussFiltSize).astype(dtype=np.float32)

    preMaskFloat = preMask.astype(np.float32)
    preMaskFloat = gaussian_filter(preMaskFloat, gaussFiltSize)

    # Change threshold.
    preMaskFloat = preMaskFloat * (0.5 / 0.95)

    # Rounding conversion.
    preMask = np.round(preMaskFloat).astype(preMask.dtype)
    preMaskFloat = preMask.astype(preMaskFloat.dtype)

    imageFloat = dst.astype(imageFloat.dtype)
    imageFloat = gaussian_filter(imageFloat, gaussFiltSize).astype(dtype=np.float32)

    # calculate first order image derivatives
    [gradientY, gradientX] = np.gradient(imageFloat)
    gradientX = gradientX.astype(dtype=np.float32)
    gradientY = gradientY.astype(dtype=np.float32)

    gradientX = np.multiply(gradientX, preMaskFloat)
    gradientY = np.multiply(gradientY, preMaskFloat)

    # matrices needed for solving linear equation system for maximizing ECC
    jacobian = np.ndarray((hs, ws * numberOfParameters), dtype=np.float32)
    hessian = np.ndarray((numberOfParameters, numberOfParameters), dtype=np.float32)
    hessianInv = np.ndarray((numberOfParameters, numberOfParameters), dtype=np.float32)
    imageProjection = np.ndarray((1, numberOfParameters), dtype=np.float32)
    templateProjection = np.ndarray((1, numberOfParameters), dtype=np.float32)
    imageProjectionHessian = np.ndarray((1, numberOfParameters), dtype=np.float32)
    errorProjection = np.ndarray((1, numberOfParameters), dtype=np.float32)

    deltaP = np.ndarray((numberOfParameters, 1))  # transformation parameter correction
    error = np.ndarray((hs, ws))  # error as 2D matrix

    # iteratively update map_matrix
    rho = -1
    last_rho = -termination_eps
    for i in range(numberOfIterations):
        if np.abs(rho - last_rho) < termination_eps:
            break

        # warp-back portion of the inputImage and gradients to the coordinate space of the templateImage
        pt = None
        if motionType != "MOTION_HOMOGRAPHY":
            tmp = np.eye(3)
            tmp[:2, :] = warpMatrix
            pt = ProjectiveTransform(matrix=tmp)
        else:
            pt = ProjectiveTransform(matrix=warpMatrix)

        imageWarped = warp(imageFloat, pt, clip=False, order=1, preserve_range=True)
        gradientXWarped = warp(gradientX, pt, clip=False, order=1, preserve_range=True)
        gradientYWarped = warp(gradientY, pt, clip=False, order=1, preserve_range=True)
        imageMask = warp(preMask, pt, clip=False, order=0, preserve_range=True)

        imgMean = np.mean(imageWarped[imageMask != 0])
        imgStd = np.std(imageWarped[imageMask != 0])
        tmpMean = np.mean(templateFloat[imageMask != 0])
        tmpStd = np.std(templateFloat[imageMask != 0])

        imageWarped[imageMask != 0] = imageWarped[imageMask != 0] - imgMean

        templateZM = np.zeros_like(templateZM)

        templateZM[imageMask != 0] = templateFloat[imageMask != 0] - tmpMean

        tmpNorm = np.sqrt(np.sum(imageMask != 0) * (tmpStd**2))
        imgNorm = np.sqrt(np.sum(imageMask != 0) * (imgStd**2))

        # calculate jacobian of image wrt parameters
        if motionType == "MOTION_AFFINE":
            jacobian = image_jacobian_affine_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, jacobian)
        elif motionType == "MOTION_HOMOGRAPHY":
            jacobian = image_jacobian_homo_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, warpMatrix, jacobian)
        elif motionType == "MOTION_TRANSLATION":
            jacobian = image_jacobian_translation_ECC(gradientXWarped, gradientYWarped, jacobian)
        elif motionType == "MOTION_EUCLIDEAN":
            jacobian = image_jacobian_euclidean_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, warpMatrix, jacobian)
        else:
            pass

        # calculate Hessian and its inverse
        hessian = project_onto_jacobian_ECC(jacobian, jacobian, hessian)

        hessianInv = np.linalg.inv(hessian)

        correlation = np.vdot(templateZM, imageWarped)

        # calculate enhanced correlation coefficient (ECC)->rho
        last_rho = rho
        rho = correlation / (imgNorm * tmpNorm)

        if np.isnan(rho):
            raise Exception("NaN encountered.")

        # project images into jacobian
        imageProjection = project_onto_jacobian_ECC(jacobian, imageWarped, imageProjection)
        templateProjection = project_onto_jacobian_ECC(jacobian, templateZM, templateProjection)

        # calculate the parameter lambda to account for illumination variation
        imageProjectionHessian = np.matmul(hessianInv, imageProjection[0])
        num = (imgNorm * imgNorm) - np.dot(imageProjection, imageProjectionHessian)
        den = correlation - np.dot(templateProjection, imageProjectionHessian)

        if den <= 0.0:
            rho = -1
            raise Exception(
                "The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped"
            )

        _lambda = num / den

        # estimate the update step delta_p
        error = _lambda * templateZM - imageWarped
        errorProjection = project_onto_jacobian_ECC(jacobian, error, errorProjection)
        deltaP = np.matmul(hessianInv, errorProjection[0])

        # update warping matrix
        warpMatrix = update_warping_matrix_ECC(warpMatrix, deltaP, motionType)

    # return final correlation coefficient
    return rho, warpMatrix
