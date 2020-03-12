import numpy as np
import random
import cvxpy as cp
import cv2
from l1_min import blackbox
from exp1 import getDCTBasis


def l1_optimize(A, Phi, B):
    n = A.shape[1]
    theta = cp.Variable(shape=(n, n))

    objective = cp.Minimize(cp.norm(theta, 1))
    constraints = [cp.matmul(A, cp.matmul(Phi, cp.matmul(theta, Phi.T))) == B]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    return theta.value


if __name__ == "__main__":
    m = 30
    data_path = "./data/"

    img_array = cv2.imread(data_path + "lena.bmp", 0)
    assert img_array.shape[0] == img_array.shape[1]
    n = img_array.shape[0]

    A = np.random.rand(m, n, n)
    B = blackbox(A, img_array)
    Phi = getDCTBasis(n)

    theta_reconstructed = l1_optimize(A, Phi, B)
    x = Phi @ theta_reconstructed @ Phi.T

    print(np.linalg.norm(x - img_array))
