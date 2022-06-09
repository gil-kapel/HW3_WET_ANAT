import copy
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from utils import calc_mse, rescale


def pyr_gen(n, m, img, gauss_pyr, laplace_pyr):
    """
    Constructs Gaussian and Laplacian pyramids out of an image.
    :param n: number of pyramid levels
    (excluding the 0th level - total number of levels - n+1).
    :param m: current pyramid level
    :param img: The input gray-scale image.
    The m-1 level of the Gaussian pyramid.
    np array of of type uint8.
    :param gauss_pyr: The Gaussian pyramid so far.
    Python list of length [m-1] containing the pyramid levels
    as np arrays of type int.
    :param laplace_pyr: The Laplacian pyramid so far.
    Python list of length [m-1] containing the pyramid levels
    as np arrays of type int.
    :return:
    gauss_pyr: The Gaussian pyramid.
    Python list of length [n-m+1] containing the pyramid levels
    as np arrays of type int.
    laplace_pyr: The Laplacian pyramid.
    Python list of length [n-m+1] containing the pyramid levels
    as np arrays of type int.
    """
    assert m >= 0 and m <= n
    # ====== YOUR CODE: ======
    # ========================
    if n == m:
        gauss_pyr.append(img)
        laplace_pyr.append(img)
        return gauss_pyr, laplace_pyr

    gauss_pyr.append(img)
    next_gauss_pyr = cv2.pyrDown(img)

    laplace_pyr.append(img.astype('int') - cv2.pyrUp(next_gauss_pyr).astype('int'))
    gauss_pyr, laplace_pyr = pyr_gen(n, m+1, next_gauss_pyr, gauss_pyr, laplace_pyr)
    return gauss_pyr, laplace_pyr


def laplace_recon(laplace_pyr):
    """
    Image reconstruction from Laplacian pyramid.
    :param laplace_pyr: The Laplacian pyramid.
    Python list containing the pyramid levels
    as np arrays of type int.
    :return:
    recon_img: The reconstructed image.
    2D np array of the same shape as laplace_pyr[0].
    """
    # ====== YOUR CODE: ======
    # ========================
    recon_img = cv2.pyrUp(laplace_pyr[-1]).astype('int')

    for i, image in enumerate(laplace_pyr[::-1][1:], start=1):
        recon_img += image
        image_range = np.max(recon_img)-np.min(recon_img)
        recon_img = (np.array((recon_img + np.abs(np.min(recon_img))) * 255 / image_range)).astype('uint8')
        if i < len(laplace_pyr) - 1:
            recon_img = cv2.pyrUp(recon_img).astype('int')
    return recon_img.astype('uint8')


def question3():
    # ---------------------------- section 3.a -------------------------------

    ironman = cv2.cvtColor(cv2.imread("../given_data/Ironman.jpg"), cv2.COLOR_BGR2GRAY)
    downey = cv2.cvtColor(cv2.imread("../given_data/Downey.jpg"), cv2.COLOR_BGR2GRAY)
    n = 4
    ironman_gauss_pyr, ironman_laplace_pyr = pyr_gen(n, 0, ironman, [], [])
    for image in ironman_gauss_pyr:
        image = np.array(image)
        cv2.imwrite(f"../my_data/q3/IronMan_Gauss_image{image.shape[0]}.jpeg", image)
    for i, image in enumerate(ironman_laplace_pyr):
        img = image
        if i != len(ironman_laplace_pyr) - 1:
            img = rescale(image)
        cv2.imwrite(f"../my_data/q3/IronMan_Laplace_image{img.shape[0]}.jpeg", img)

    downey_gauss_pyr, downey_laplace_pyr = pyr_gen(n, 0, downey, [], [])
    for image in downey_gauss_pyr:
        image = np.array(image)
        cv2.imwrite(f"../my_data/q3/Downey_Gauss_image{image.shape[0]}.jpeg", image)
    for i, image in enumerate(downey_laplace_pyr):
        img = image
        if i != len(downey_laplace_pyr) - 1:
            img = rescale(image)
        cv2.imwrite(f"../my_data/q3/Downey_Laplace_image{img.shape[0]}.jpeg", img)

    # ---------------------------- section 3.b -------------------------------
    ironman_recon = laplace_recon(copy.deepcopy(ironman_laplace_pyr))
    downey_recon = laplace_recon(copy.deepcopy(downey_laplace_pyr))
    ironman_mse = calc_mse(ironman_recon, ironman)
    downey_mse = calc_mse(downey_recon, downey)
    cv2.imshow(f'ironman_recon with mse:{ironman_mse}', ironman_recon)
    cv2.imshow(f'downey_recon with mse:{downey_mse}', downey_recon)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # ---------------------------- section 3.c -------------------------------
    mask = np.zeros(downey.shape)
    mid_img = int(mask.shape[0]/2)
    mask[:, :mid_img] = 1
    mask_gauss_pyr, mask_laplace_pyr = pyr_gen(n, 0, mask, [], [])
    blend_laplace = [(ironman_laplace_pyr[i] * mask_gauss_pyr[i] +
                     (1 - (mask_gauss_pyr[i])) * downey_laplace_pyr[i]).astype('int')
                     for i in range(len(ironman_laplace_pyr))]
    blend_laplace[-1] = blend_laplace[-1].astype('uint8')
    blend_recon = laplace_recon(copy.deepcopy(blend_laplace))
    blend_mse = calc_mse(blend_recon, downey)
    cv2.imshow(f'blend_mse with mse:{blend_mse}', blend_recon)


if __name__ == '__main__':
    question3()
    cv2.destroyAllWindows()

