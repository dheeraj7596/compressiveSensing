import numpy as np
import cv2
import matplotlib.pyplot as plt
from l1_min import blackbox, l1_optimize
from exp1 import getDCTBasis

if __name__ == "__main__":
    data_path = "./data/"

    img_array = cv2.imread(data_path + "lena.bmp", 0)
    plt.figure()
    plt.title("Original Image")
    plt.imshow(img_array, cmap='gray')
    plt.show()
    assert img_array.shape[0] == img_array.shape[1]
    n = img_array.shape[0]
    Phi = getDCTBasis(n)

    theta_orig = Phi.T @ img_array @ Phi
    theta_reconstructed = np.array([])
    for col in theta_orig.T:
        col = col.reshape((n, 1))
        A = np.random.rand(n, n)
        B = blackbox(A, col)
        col_reconstructed = l1_optimize(A, B)
        if len(theta_reconstructed) == 0:
            theta_reconstructed = col_reconstructed
        else:
            theta_reconstructed = np.hstack((theta_reconstructed, col_reconstructed))

    print("Theta reconstruction error: ", np.linalg.norm(theta_reconstructed - theta_orig))

    img_array_reconstructed = Phi @ theta_reconstructed @ Phi.T
    plt.figure()
    plt.title("Reconstructed Image")
    plt.imshow(img_array_reconstructed, cmap='gray')
    plt.show()
    print("Image reconstruction error: ", np.linalg.norm(img_array_reconstructed - img_array))
