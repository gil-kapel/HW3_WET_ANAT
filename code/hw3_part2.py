import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
from utils import rescale, calc_mse, show_single_graph


def question2():
    # ---------------------------- section 2.a -------------------------------
    # --------------------------------------------- section 2.1 --------------
    image_list = []
    os.chdir("../given_data/LFW")
    for filename in glob.glob("*.pgm"):
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image_list.append(image)
    image_list = np.array(image_list)
    chosen_img = np.stack([image_list[10], image_list[100], image_list[1000], image_list[10000]])
    for i, image in enumerate(chosen_img, start=1):
        cv2.imshow(f'Image {i}', image)

    # --------------------------------------------- section 2.2 -------------
    X = np.array([image_list[i].flatten('F') for i in range(len(image_list))]).T

    # --------------------------------------------- section 2.3 -------------
    avg_image = np.mean(X, axis=1).astype('uint8')
    image_shape = (image_list.shape[1], image_list.shape[2])
    mat_avg_image = avg_image.reshape(image_shape, order='F')
    cv2.imshow('Average image', mat_avg_image)
    mu = avg_image.reshape((avg_image.shape[0], 1)).astype('int')
    Y = X - mu
    y_cov = np.cov(Y)

    # ---------------------------- section 2.b -------------------------------
    # --------------------------------------------- section 2.1 --------------
    eig_vals, eig_vecs = np.linalg.eigh(y_cov)

    k = 10
    ten_eig_vals, ten_eig_vecs = eig_vals[::-1][:k], eig_vecs[:, ::-1][:, :k]
    show_single_graph(ten_eig_vals, 'Ten largest eigen values', 'index','eigen value')

    for i, vector in enumerate(ten_eig_vecs.T[:4]):
        vector_int = rescale(vector)
        cv2.imshow(f'Eigen image_{i+1}', vector_int.reshape(image_shape, order='F'))

    # ---------------------------- section 2.c -------------------------------
    P = ten_eig_vecs.T @ Y

    # ---------------------------- section 2.d -------------------------------
    p_i = np.stack([P[:, 10], P[:, 100], P[:, 1000], P[:, 10000]]).T
    x_hat = ten_eig_vecs @ p_i + mu
    for i, vector in enumerate(x_hat.T):
        image = vector.reshape(image_shape, order='F').astype('uint8')
        mse = calc_mse(image, chosen_img[i])
        cv2.imshow(f'{k} compression, MSE= {mse}', image)

    # ---------------------------- section 2.e -------------------------------
    k = 570
    eig_vals_e, eig_vecs_e = eig_vals[::-1][:k], eig_vecs[:, ::-1][:, :k]
    P = eig_vecs_e.T @ Y
    p_i = np.stack([P[:, 10], P[:, 100], P[:, 1000], P[:, 10000]]).T
    x_hat = eig_vecs_e @ p_i + mu
    for i, vector in enumerate(x_hat.T):
        image = vector.reshape(image_shape, order='F').astype('uint8')
        mse = calc_mse(image, chosen_img[i])
        cv2.imshow(f'{k} compression, MSE= {mse}', image)


if __name__ == '__main__':
    question2()
    cv2.destroyAllWindows()
