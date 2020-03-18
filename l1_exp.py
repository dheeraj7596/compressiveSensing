# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:37:02 2020

@author: 14uda
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from l1_min import l1_optimize_with_noise
from OMP import rmse
if __name__ == "__main__":
    n = 200

    s_arr = [10, 30]
    m_arr = [40, 60, 80, 100, 120]
    var_arr = [0.00, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2]  ##Relative noise intensity
    num_trials = 50

    results = np.zeros((len(s_arr), len(m_arr), len(var_arr), num_trials))

    for i, s in enumerate(s_arr):
        for j, m in enumerate(m_arr):
            print(j)
            for k, sigma in enumerate(var_arr):
                for l in range(num_trials):
                    indices = np.random.choice(n, s, replace=False)
                    theta = np.zeros((n, 1))
                    theta[indices, :] = np.random.randn(s, 1)
                    theta = theta / np.linalg.norm(2)
                    A = np.random.rand(m, n)
                    y = A @ theta + sigma * np.random.randn(m, 1)
                    abs_tol = math.sqrt(4 * n) * sigma * np.linalg.norm(theta)
                    theta_recon = l1_optimize_with_noise(A, y, sigma)
                    results[i, j, k, l] = rmse(theta, theta_recon)

    results_median = np.median(results, axis=-1)
    for i, s in enumerate(s_arr):
        plt.figure()
        plt.title("Sparsity level: " + str(s))
        im = plt.imshow(results_median[i, :, :])
        ax = plt.gca()
        ax.set_xticks(np.arange(len(var_arr)))
        ax.set_yticks(np.arange(len(m_arr)))
    # ... and label them with the respective list entries.
        ax.set_xticklabels(var_arr*100)
        ax.set_yticklabels(m_arr)
        ax.set_xlabel("Relative Intensity of Noise")
        ax.set_ylabel("Number of measurements")
        for i2 in range(len(m_arr)):
            for i3 in range(len(var_arr)):
                text = ax.text(i3, i2, round(results_median[i,i2,i3],2),
                               ha="center", va="center", color="w")
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Relative Mean Squared error", rotation=-90, va="bottom")
        plt.show()
