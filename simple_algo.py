import numpy as np
import random
import cvxpy as cp


def blackbox(A, x_orig):
    return np.matmul(A, x_orig)


if __name__ == "__main__":
    s = 5
    m = 30
    n = 200
    x_orig = np.random.uniform(low=0.5, high=100.3, size=(n, 1))
    indices = random.sample(range(0, len(x_orig)), s)

    for i, p in enumerate(x_orig):
        if i in indices:
            continue
        x_orig[i] = 0

    A = np.random.rand(m, n)
    B = blackbox(A, x_orig)

    x = cp.Variable(shape=(n, 1))

    objective = cp.Minimize(cp.norm(x, 1))
    constraints = [cp.matmul(A, x) == B]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    print(np.linalg.norm(x.value - x_orig))
