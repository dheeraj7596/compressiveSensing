import cv2
import numpy as np
from scipy.fftpack import idct, dct
import matplotlib.pyplot as plt


def getDCTBasis(n):
    '''Gives us the basis for 1d-DCT for n-dimensional signals'''
    return dct(np.eye(n), type=2, norm='ortho')


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


if __name__ == "__main__":
    data_path = "./data/"
    img_array = cv2.imread(data_path + "lena.bmp", 0)
    assert img_array.shape[0] == img_array.shape[1]
    n = img_array.shape[0]
    # img = Image.fromarray(img_array)
    shape = img_array.shape
    plt.figure()
    plt.title("Original Image")
    plt.imshow(img_array, cmap='gray')
    plt.show()

    theta = dct2(img_array)
    Phi = getDCTBasis(n)
    theta2 = Phi.T @ img_array @ Phi

    U = np.absolute(theta).flatten()

    U_sorted = sorted(U, reverse=True)
    plt.figure()
    plt.title("DCT2 coefficients")
    plt.plot(range(len(U_sorted)), U_sorted)
    plt.show()

    plt.figure()
    s = 20000
    plt.title('Reconstructed with {0} coefficients out of {1}'.format(s, n * n))
    thresh = U_sorted[s]
    theta[np.absolute(theta) < thresh] = 0
    X_cap = idct2(theta)
    plt.imshow(X_cap, cmap='gray')
    plt.show()
    pass
