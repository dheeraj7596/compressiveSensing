import numpy as np
import cv2
import matplotlib.pyplot as plt
from l1_min import blackbox, l1_optimize, l1_optimize_with_noise
from exp1 import getDCTBasis


def get_theta_orig(Phi, img_array):
    theta_orig = np.array([])
    for col in img_array.T:
        col = col.reshape((n, 1))
        theta_col = Phi.T @ col
        if len(theta_orig) == 0:
            theta_orig = theta_col
        else:
            theta_orig = np.hstack((theta_orig, theta_col))
    return theta_orig


def reconstruct_img(Phi, theta_reconstructed):
    img_array_reconstructed = np.array([])
    for col in theta_reconstructed.T:
        col = col.reshape((n, 1))
        img_col = Phi @ col
        if len(img_array_reconstructed) == 0:
            img_array_reconstructed = img_col
        else:
            img_array_reconstructed = np.hstack((img_array_reconstructed, img_col))
    return img_array_reconstructed


if __name__ == "__main__":
    data_path = "./data/"

    m = 50
    noise_variance = 15
    img_array = cv2.imread(data_path + "lena.bmp", 0)
    plt.figure()
    plt.title("Original Image")
    plt.imshow(img_array, cmap='gray')
    plt.show()
    assert img_array.shape[0] == img_array.shape[1]
    n = img_array.shape[0]
    Phi = getDCTBasis(n)

    theta_orig = get_theta_orig(Phi, img_array)
    theta_reconstructed = np.array([])
    for col in img_array.T:
        col = col.reshape((n, 1))
        A = np.random.rand(m, n)
        B = blackbox(A, col)
        col_reconstructed = l1_optimize(A @ Phi, B)
        # col_reconstructed = l1_optimize_with_noise(A, B, noise_variance)
        if len(theta_reconstructed) == 0:
            theta_reconstructed = col_reconstructed
        else:
            theta_reconstructed = np.hstack((theta_reconstructed, col_reconstructed))

    print("Theta reconstruction error: ", np.linalg.norm(theta_reconstructed - theta_orig))

    img_array_reconstructed = reconstruct_img(Phi, theta_reconstructed)
    plt.figure()
    plt.title("Reconstructed Image")
    plt.imshow(img_array_reconstructed, cmap='gray')
    plt.show()
    print("Image reconstruction error: ", np.linalg.norm(img_array_reconstructed - img_array))
