import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
from utils import show_single_graph, calc_mse


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
    avg_image = np.mean(X, axis=1)
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
    ten_eig_vals, ten_eig_vecs = eig_vals[::-1][:k], eig_vecs[::-1][:k]
    x = np.linspace(0, k, k)
    plt.plot(x, ten_eig_vals)
    plt.title('Eigen values')
    plt.show()

    for i, vector in enumerate(ten_eig_vecs[:4]):
        vector_range = np.max(vector) - np.min(vector)
        vector_int = ((vector + np.abs(np.min(vector))) * 255 / vector_range).astype('uint8')
        cv2.imshow(f'Eigen image_{i+1}', vector_int.reshape(image_shape, order='F'))

    # ---------------------------- section 2.c -------------------------------
    P = ten_eig_vecs @ Y

    # ---------------------------- section 2.d -------------------------------
    p_i = P[:, :4]
    x_hat = ten_eig_vecs.T @ p_i + mu
    x_hat = x_hat.reshape((image_shape[0], image_shape[1], 4), order='F')
    x_hat = np.swapaxes(x_hat, 2, 1)
    x_hat = np.swapaxes(x_hat, 1, 0)
    x_b = image_list[:4]
    for i, vector in enumerate(x_b[:4]):
        mse = calc_mse(vector, x_hat[i])
        cv2.imshow(f'restored image, MSE= {mse}', vector)
    pass

    # ---------------------------- section 2.e -------------------------------
    k = 570
    eig_vals_e, eig_vecs_e = eig_vals[::-1][:k], eig_vecs[::-1][:k]
    P_e = eig_vecs_e @ Y
    p_i_e = P_e[:, :4]
    x_hat_e = eig_vecs_e.T @ p_i_e + mu
    x_hat_e = x_hat_e.reshape((image_shape[0], image_shape[1], 4), order='F')
    x_hat_e = np.swapaxes(x_hat_e, 2, 1)
    x_hat_e = np.swapaxes(x_hat_e, 1, 0)
    for i, vector in enumerate(x_b[:4]):
        mse_e = calc_mse(vector, x_hat_e[i])
        cv2.imshow(f'restored image, MSE= {mse_e}', vector)


if __name__ == '__main__':
    question2()
    cv2.destroyAllWindows()
