# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:10:33 2020

@author: 14uda
"""

import numpy as np
from scipy.fftpack import idct, dct

x = np.random.randn((100))

theta = dct(x, type=2, norm='ortho')

Phi = dct(np.eye(100), type=2, norm='ortho')

theta2 = Phi.T @ x
