from PIL import Image
import numpy as np
from scipy.fftpack import idct, dct
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_path = "./data/"
    img = Image.open(data_path + "lena.bmp")
    img_array = np.asarray(img)/255
    shape = img_array.shape
    plt.imshow(img,cmap='gray')
    
    
    theta = dct(img_array, type=2)
    
    img_back = idct(dct(img_array))
    plt.figure()
    plt.imshow(img_back,cmap='gray')
    U = np.absolute(theta).flatten()

    U_sorted = sorted(U, reverse=True)
    plt.figure()
    plt.plot(range(len(U_sorted)), U_sorted)
    plt.show()

    plt.figure()
    thresh = U_sorted[200000]
    theta[np.absolute(theta) < thresh] = 0
    X_cap = idct(theta, type=2)
    im = Image.fromarray(X_cap)
    plt.imshow(im,cmap='gray')
    pass
