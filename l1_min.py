import numpy as np
import random
import cvxpy as cp


def blackbox(A, x_orig):
    return np.matmul(A, x_orig)


def l1_optimize(A, B):
    n = A.shape[1]
    x = cp.Variable(shape=(n, 1))

    objective = cp.Minimize(cp.norm(x, 1))
    constraints = [cp.matmul(A, x) == B]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    return x.value


if __name__ == "__main__":
    s = 5
    m = 30
    n = 200
    indices = np.random.choice(n, s, replace=False)
    theta = np.zeros((n, 1))
    theta[indices, :] = np.random.randn(s, 1)

    A = np.random.rand(m, n)
    B = blackbox(A, theta)

    x = l1_optimize(A, B)

    print(np.linalg.norm(x - theta))