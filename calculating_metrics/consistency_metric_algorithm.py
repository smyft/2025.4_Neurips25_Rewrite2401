from itertools import product
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.optimize import linprog


def calculate_consistency_metric(phi, N_A, L):
    """
    for a joint distribution of posterior phi, calculate consistency metric
    Args:
        phi (dict): joint distribution of posterior beliefs, map a joint posterior to its probability
        N_A (int): # of agents
        L (int): # of world states
    return:
        consistency metric
    """
    # preparing part

    # we want to find all possible f_h
    # be careful different agent h may have same f_h
    f_h2phi_fh = [dict() for i in range(N_A)] # N_A dicts. Every dicts map one possible f_h to its prob
    for f in phi.keys():
        f = np.frombuffer(f, dtype=np.float32).reshape(N_A, L)    # joint posterior
        for h in range(N_A):
            f_h = f[h]  # agent h's posterior
            f_h2phi_fh[h][f_h.tobytes()] = f_h2phi_fh[h].get(f_h.tobytes(), 0) + phi[f.tobytes()]
    
    # different agent h may have same f_h, so we can't use f_h as a dict key
    fh_2_idx = dict()   # map f_h to idx in list_all_fh, as same as the idx in trade pattern x
    n_fh = 0
    for h in range(N_A):
        for fh in f_h2phi_fh[h].keys():
            fh_2_idx[(h, fh)] = n_fh
            n_fh += 1
    '''
    print("------")
    print("list all possible f_h and its idx")
    for h, fh in fh_2_idx.keys():
        print(f"agent: {h}, f_h: {np.frombuffer(fh, dtype=np.float32)}, idx: {fh_2_idx[(h, fh)]}")
    print("------")
    '''

    # linear programming part

    # modify the origin theorem
    # introduce auxiliary variables y to deal with the max operator
    # y: f->R, correspond to max_theta(\sum_x_i)

    B = np.zeros((n_fh, n_fh * L))
    for h, fh in fh_2_idx.keys():
        idx = fh_2_idx[(h, fh)]
        B[idx][idx * L: (idx + 1) * L] = np.frombuffer(fh, dtype=np.float32)

    # print("------")
    # print("B:")
    # print(B)
    # print("------")

    n_f = len(phi.keys())
    C = np.zeros((n_f * L, n_fh * L))
    for f_idx, f in enumerate(phi.keys()):
        f = np.frombuffer(f, dtype=np.float32).reshape(N_A, L)    # joint posterior
        for theta in range(L):
            for h in range(N_A):
                f_h = f[h]
                x_idx = fh_2_idx[(h, f_h.tobytes())]
                C[f_idx * L + theta][x_idx * L + theta] = 1
    # print("------")
    # print("C:")
    # print(C)
    # print("------")

    D = np.zeros((n_f * L, n_f))
    for f_idx in range(n_f):
        for theta in range(L):
            D[f_idx * L + theta][f_idx] = 1
    # print("------")
    # print("D:")
    # print(D)
    # print("------")

    A = np.zeros((n_fh+n_f*L+2*n_fh*L,4*n_fh*L+2*n_f+n_f*L))
    A[0:n_fh, 0:n_fh*L] = B
    A[0:n_fh, n_fh*L:2*n_fh*L] = -B
    A[n_fh:n_fh+n_f*L, 0:n_fh*L] = C
    A[n_fh:n_fh+n_f*L, n_fh*L:2*n_fh*L] = -C
    A[n_fh:n_fh+n_f*L, 2*n_fh*L:2*n_fh*L+n_f] = -D
    A[n_fh:n_fh+n_f*L, 2*n_fh*L+n_f:2*n_fh*L+2*n_f] = D
    A[n_fh:n_fh+n_f*L, 2*n_fh*L+2*n_f:2*n_fh*L+2*n_f+n_f*L] = np.eye(n_f*L)
    A[n_fh+n_f*L:n_fh+n_f*L+n_fh*L, 0:n_fh*L] = np.eye(n_fh*L)
    A[n_fh+n_f*L:n_fh+n_f*L+n_fh*L, n_fh*L:2*n_fh*L] = -np.eye(n_fh*L)
    A[n_fh+n_f*L:n_fh+n_f*L+n_fh*L, 2*n_fh*L+2*n_f+n_f*L:3*n_fh*L+2*n_f+n_f*L] = np.eye(n_fh*L)
    A[n_fh+n_f*L+n_fh*L:n_fh+n_f*L+2*n_fh*L, 0:n_fh*L] = -np.eye(n_fh*L)
    A[n_fh+n_f*L+n_fh*L:n_fh+n_f*L+2*n_fh*L, n_fh*L:2*n_fh*L] = np.eye(n_fh*L)
    A[n_fh+n_f*L+n_fh*L:n_fh+n_f*L+2*n_fh*L, 3*n_fh*L+2*n_f+n_f*L:4*n_fh*L+2*n_f+n_f*L] = np.eye(n_fh*L)
    # print("------")
    # print("A:")
    # np.set_printoptions(threshold=np.inf)
    # print(A)
    # print("------")

    b = np.zeros((n_fh+n_f*L+2*n_fh*L,))
    b[n_fh+n_f*L:n_fh+n_f*L+2*n_fh*L] = np.ones((2*n_fh*L,))
    # print("------")
    # print("b:")
    # print(b)
    # print("------")

    v_p = np.zeros((n_f,))
    for f_idx, f in enumerate(phi.keys()):
        v_p[f_idx] = phi[f]
    # print("------")
    # print("v_p:")
    # print(v_p)
    # print("------")

    c = np.zeros((4*n_fh*L+2*n_f+n_f*L,))
    c[2*n_fh*L:2*n_fh*L+n_f] = -v_p
    c[2*n_fh*L+n_f:2*n_fh*L+2*n_f] = v_p
    # print("------")
    # print("c:")
    # print(c)
    # print("------")
    
    bounds = [(0, None) for _ in range(4*n_fh*L+2*n_f+n_f*L)]

    res = linprog(-c, A_eq=A, b_eq=b, bounds=bounds)
    consistency_metric = res.fun

    assert consistency_metric is not None, "linear programming returns None"

    return consistency_metric


if __name__=="__main__":
    '''
    label_nums = 2
    N_A = 2

    r = 3 / 4
    c = 1 / 4
    f1 = np.array([[1 - r, r], [1 - r, r]], dtype=np.float32)
    f2 = np.array([[r, 1 - r], [r, 1 - r]], dtype=np.float32)
    f3 = np.array([[1 - r, r], [r, 1 - r]], dtype=np.float32)
    f4 = np.array([[r, 1 - r], [1 - r, r]], dtype=np.float32)
    # phi1 = phi2 = (r ** 2 + (1 - r) ** 2) / 2
    # phi3 = phi4 = r * (1 - r)
    phi1 = phi2 = c / 2
    phi3 = phi4 = 1 / 2 - c / 2
    phi = {}
    phi[f1.tobytes()] = phi1
    phi[f2.tobytes()] = phi2
    phi[f3.tobytes()] = phi3
    phi[f4.tobytes()] = phi4

    print("------")
    print("joint posterior beliefs distribution: phi")
    for k in phi.keys():
        print(f"f: {np.frombuffer(k, dtype=np.float32).reshape(N_A, label_nums)}")
        print(f"phi(f): {phi[k]}")
    print("------")

    consistency_metric = calculate_consistency_metric(phi, N_A, label_nums)
    print(f"consistency metric: {consistency_metric}")
    '''
    
    label_nums = 2
    N_A = 2

    metrics = []
    r = 3/4
    range_c = np.linspace(0, 1, 2**10+1)

    for c in range_c:
        f1 = np.array([[1 - r, r], [1 - r, r]], dtype=np.float32)
        f2 = np.array([[r, 1 - r], [r, 1 - r]], dtype=np.float32)
        f3 = np.array([[1 - r, r], [r, 1 - r]], dtype=np.float32)
        f4 = np.array([[r, 1 - r], [1 - r, r]], dtype=np.float32)
        phi1 = phi2 = c / 2
        phi3 = phi4 = 1 / 2 - c / 2
        phi = {}
        phi[f1.tobytes()] = phi1
        phi[f2.tobytes()] = phi2
        phi[f3.tobytes()] = phi3
        phi[f4.tobytes()] = phi4

        metric = calculate_consistency_metric(phi, N_A, label_nums)
        metrics.append(metric)
        # print(f"consistency metric: {metric}")

    plt.figure()
    plt.plot(range_c, metrics)
    plt.xlabel('c')
    plt.ylabel('metric')
    plt.savefig('result_images/example.png')
    plt.show()
    