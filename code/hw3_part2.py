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
    for i, image in enumerate(image_list):
        if i > 3:
            break
        cv2.imshow(f'Image {i}', image)

    # --------------------------------------------- section 2.2 -------------
    column_stack_image = image_list.reshape((image_list.shape[1] * image_list.shape[2], image_list.shape[0]), order='F')

    # --------------------------------------------- section 2.3 -------------
    avg_image = np.mean(column_stack_image, axis=1)
    image_shape = (image_list.shape[1], image_list.shape[2])
    mat_avg_image = avg_image.reshape(image_shape, order='F')
    cv2.imshow('Average image', mat_avg_image)
    mu = avg_image.reshape((avg_image.shape[0], 1))
    y = column_stack_image - mu
    y_cov = np.cov(y)
    cv2.destroyAllWindows()

    # ---------------------------- section 2.b -------------------------------
    # --------------------------------------------- section 2.1 --------------
    eig_vals, eig_vecs = np.linalg.eigh(y_cov)
    eig_vals, eig_vecs = eig_vals[::-1][:10], eig_vecs[::-1][:10]
    x = np.linspace(0, 10, 10)
    plt.plot(x, eig_vals)
    plt.title('Eigen values')
    plt.show()
    plt.close()
    for i, vector in enumerate(eig_vecs):
        if i > 3:
            break
        cv2.imshow('Eigen image', vector.reshape(image_shape, order='F'))

    # ---------------------------- section 2.c -------------------------------
    P = eig_vecs @ y[:, :4]

    # ---------------------------- section 2.d -------------------------------
    x_hat = eig_vecs.T @ P + mu
    x_b = image_list[:4].reshape(x_hat.shape, order='F')
    mse = calc_mse(x_b, x_hat)
    for i, vector in enumerate(x_b.T):
        if i > 3:
            break
        cv2.imshow(f'restored image, MSE= {mse[i]}', vector.reshape(image_shape, order='F'))
    pass

    # ---------------------------- section 2.e -------------------------------
    eig_vals_e, eig_vecs_e = eig_vals[::-1][:10], eig_vecs[::-1][:10]
    P_e = eig_vecs_e @ y[:, :4]
    x_hat_e = eig_vecs.T @ P + mu
    x_b_e= image_list[:4].reshape(x_hat_e.shape, order='F')
    mse_e = ((x_hat_e - x_b_e) ** 2).mean(axis=0).mean(axis=0)
    for i, vector in enumerate(x_b_e.T):
        if i > 3:
            break
        cv2.imshow(f'restored image, MSE= {mse_e[i]}', vector.reshape(image_shape, order='F'))


if __name__ == '__main__':
    question2()
    cv2.destroyAllWindows()
