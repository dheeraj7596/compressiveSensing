# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:53:26 2020

@author: 14uda
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from OMP import OMP,rmse
from exp1 import getDCTBasis
import math

if __name__ == "__main__":
    data_path = "./data/"

    sigma = 0.01
    img_array = cv2.imread(data_path + "clown.bmp", 0)
    plt.figure()
    plt.title("Original Image")
    plt.imshow(img_array, cmap='gray')
    plt.show()
    assert img_array.shape[0] == img_array.shape[1]
    n = img_array.shape[0]
    Phi = getDCTBasis(n)
    m = 300
    
    theta = Phi.T @ img_array  # Id DCT of columns
    theta_recon = np.zeros((n,n))
    
    for i in range(n):
        col = np.expand_dims(theta[:,i],axis=1)
        A = np.random.rand(m, n) @ Phi
        y = A @ col + sigma*np.linalg.norm(col)*np.random.randn(m, 1)
        abs_tol = math.sqrt(4 * n) * sigma * np.linalg.norm(col)
        col_recon = OMP(A, y, abs_tol)
        theta_recon[:,i] = col_recon.squeeze()
    img_recon = Phi @ theta_recon # 1D IDCT of cols
    plt.figure()
    plt.title("Reconstructed Image")
    plt.imshow(img_recon, cmap='gray')
    plt.show()
    print("Image reconstruction error: ",rmse(theta,theta_recon))