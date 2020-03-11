# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:10:27 2020

@author: 14uda
"""

import numpy as np


def rmse(x, x_recon):
    return np.sum((x - x_recon) ** 2) / np.sum(x ** 2)


def OMP(A, y, abs_tol=1e-6, max_iter=1000):
    m, n = A.shape
    r = y.copy()
    theta = np.zeros((n, 1))
    T = np.zeros((n), dtype=bool)
    A_norm = A / np.sqrt(np.sum(A ** 2, axis=0))
    for itr in range(max_iter):
        index = np.argmax(np.abs(r.T @ A_norm))
        T[index] = True
        A_T = A[:, T]
        theta[T, :] = np.linalg.lstsq(A_T, y, rcond=None)[0]
        r = y - A_T @ theta[T, :]

        if (np.linalg.norm(r) < abs_tol):
            # print(np.linalg.norm(r))
            break
    return theta


s = 5
n = 200
m = 20

indices = np.random.choice(n, s, replace=False)
theta = np.zeros((n, 1))
theta[indices, :] = np.random.randn(s, 1)

A = np.random.rand(m, n)

y = A @ theta

theta_recon = OMP(A, y)
# print(np.hstack((theta,theta_recon)))
print(rmse(theta, theta_recon))
