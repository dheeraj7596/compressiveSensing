# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:24:43 2020

@author: 14uda
"""

import numpy as np


def rmse(x, x_recon):
    return np.sum((x - x_recon) ** 2) / np.sum(x ** 2)


def GRASP(A, y, s, abs_tol=1e-6, max_iter=100):
    m, n = A.shape
    abs_tol = max(abs_tol,1e-6)
    r = y.copy()
    theta = np.zeros((n, 1))
    T = np.zeros((n), dtype=bool)
    for itr in range(max_iter):
        gradient = A.T @ r
        indices = np.argpartition(np.squeeze(np.abs(gradient)), -3 * s, axis=0)[-3 * s:]
        T[indices] = True
        A_T = A[:, T]
        theta[T, :] = np.linalg.lstsq(A_T, y, rcond=None)[0]

        ##Prune estimate
        indices = np.argpartition(np.squeeze(np.abs(theta)), -s)[-s:]
        T = np.zeros((n), dtype=bool)
        T[indices] = True
        theta[~T] = 0
        A_T = A[:, T]
        r = y - A_T @ theta[T, :]

        if (np.linalg.norm(r) < abs_tol):
            # print(np.linalg.norm(r))
            break
    return theta


s = 5
n = 200
m = 30

indices = np.random.choice(n, s, replace=False)
theta = np.zeros((n, 1))
theta[indices, :] = np.random.randn(s, 1)

A = np.random.rand(m, n)

y = A @ theta

theta_recon = GRASP(A, y, s)
# print(np.stack((theta,theta_recon)))
print(rmse(theta, theta_recon))
