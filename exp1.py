import cv2
import numpy as np
from scipy.fftpack import idct, dct
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_path = "./data/"
    img_array = cv2.imread(data_path + "lena.bmp", 0)
    # img = Image.fromarray(img_array)
    shape = img_array.shape
    plt.figure()
    plt.title("Original Image")
    plt.imshow(img_array, cmap='gray')
    plt.show()

    theta = dct(img_array, type=2, norm="ortho")

    img_back = idct(dct(img_array, norm="ortho"), norm="ortho")
    plt.figure()
    plt.title('Reconstructed with all coeff')
    plt.imshow(img_back, cmap='gray')
    plt.show()

    U = np.absolute(theta).flatten()

    U_sorted = sorted(U, reverse=True)
    plt.figure()
    plt.title("DCT coefficients")
    plt.plot(range(len(U_sorted)), U_sorted)
    plt.show()

    plt.figure()
    plt.title('Reconstructed with some coeff')
    thresh = U_sorted[22000]
    theta[np.absolute(theta) < thresh] = 0
    X_cap = idct(theta, type=2)
    plt.imshow(X_cap, cmap='gray')
    plt.show()
    pass
