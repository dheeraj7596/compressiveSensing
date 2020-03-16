# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:05:22 2020

@author: 14uda
"""

from GRASP import GRASP
from OMP import rmse
import numpy as np
import math

n = 200

s_arr = [10, 20, 40]
m_arr = [40, 50, 70, 80, 100]
var_arr = [0.00,0.01, 0.03, 0.05, 0.07, 0.1, 0.2] ##Relative noise intensity
num_trials = 10

results = np.zeros((len(s_arr),len(m_arr),len(var_arr),num_trials))

for i,s in enumerate(s_arr):
    for j,m in enumerate(m_arr):
        print(j)
        for k,sigma in enumerate(var_arr):
            for l in range(num_trials):
                indices = np.random.choice(n, s, replace=False)
                theta = np.zeros((n, 1))
                theta[indices, :] = np.random.randn(s, 1)
                A = np.random.rand(m, n)
                y = A@theta + sigma*np.linalg.norm(theta)*np.random.randn(m,1)
                abs_tol = math.sqrt(4*n) * sigma*np.linalg.norm(theta)
                theta_recon = GRASP(A,y,s,abs_tol)
                results[i,j,k,l] = rmse(theta,theta_recon)

## TODO: plot the reults

print(results)