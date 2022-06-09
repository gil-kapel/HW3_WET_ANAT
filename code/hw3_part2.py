import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
from utils import rescale, calc_mse


def question2():
    # ---------------------------- section 2.a -------------------------------
    # --------------------------------------------- section 2.1 --------------
    image_list = []
    os.chdir("../given_data/LFW")
    for filename in glob.glob("*.pgm"):
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image_list.append(image)
    image_list = np.array(image_list)
    for i, image in enumerate(image_list[:4]):
        cv2.imshow(f'Image {i}', image)

    # --------------------------------------------- section 2.2 -------------
    X = image_list.reshape((image_list.shape[1] * image_list.shape[2], image_list.shape[0]), order='F')

    # --------------------------------------------- section 2.3 -------------
    avg_image = np.mean(X, axis=1).astype('uint8')
    image_shape = (image_list.shape[1], image_list.shape[2])
    mat_avg_image = avg_image.reshape(image_shape, order='F')
    cv2.imshow('Average image', mat_avg_image)
    mu = avg_image.reshape((avg_image.shape[0], 1))
    Y = X - mu
    y_cov = np.cov(Y)

    # ---------------------------- section 2.b -------------------------------
    # --------------------------------------------- section 2.1 --------------
    k = 10
    eig_vals, eig_vecs = np.linalg.eigh(y_cov)
    ten_eig_vals, ten_eig_vecs = eig_vals[::-1][:k], eig_vecs[:, ::-1]
    ten_eig_vecs = ten_eig_vecs[:, :k]
    x = np.linspace(0, k, k)
    plt.plot(x, ten_eig_vals)
    plt.title('Eigen values')
    plt.show()

    nine_eig_vals = eig_vals[::-1][1:k]
    x = np.linspace(1, k, k-1)
    plt.plot(x, nine_eig_vals)
    plt.title('Nine first eigen values, without the highest')
    plt.show()
    for i, vector in enumerate(ten_eig_vecs.T[:4]):
        vector_int = rescale(vector)
        cv2.imshow(f'Eigen image_{i+1}', vector_int.reshape(image_shape, order='F'))

    # ---------------------------- section 2.c -------------------------------
    P = ten_eig_vecs.T @ Y

    # ---------------------------- section 2.d -------------------------------
    p_i = P[:, :4]
    x_hat = ten_eig_vecs @ p_i + mu
    for i, vector in enumerate(x_hat.T):
        image = rescale(vector.reshape(image_shape, order='F'))
        mse = calc_mse(image, image_list[i])
        cv2.imshow(f'{k} compression, MSE= {mse}', image)

    # ---------------------------- section 2.e -------------------------------
    k = 570
    eig_vals_e, eig_vecs_e = eig_vals[::-1][:k], eig_vecs[:, ::-1]
    eig_vecs_e = eig_vecs_e[:, :k]
    P = eig_vecs_e.T @ Y
    p_i = P[:, :4]
    x_hat = eig_vecs_e @ p_i + mu
    for i, vector in enumerate(x_hat.T):
        image = rescale(vector.reshape(image_shape, order='F'))
        mse = calc_mse(image, image_list[i])
        cv2.imshow(f'{k} compression, MSE= {mse}', image)


if __name__ == '__main__':
    question2()
    cv2.destroyAllWindows()
