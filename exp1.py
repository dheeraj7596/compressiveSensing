from PIL import Image
import numpy as np
from scipy.fftpack import idct, dct
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_path = "./data/"
    img = Image.open(data_path + "lena.bmp")
    img_array = np.asarray(img)
    shape = img_array.shape

    theta = idct(img_array, type=2)
    U = np.absolute(theta).flatten()

    U_sorted = sorted(U, reverse=True)
    plt.figure()
    plt.plot(range(len(U_sorted)), U_sorted)
    plt.show()

    thresh = U_sorted[15000]
    theta[np.absolute(theta) < thresh] = 0
    X_cap = dct(theta, type=2)
    im = Image.fromarray(X_cap)
    im.show()

    pass
