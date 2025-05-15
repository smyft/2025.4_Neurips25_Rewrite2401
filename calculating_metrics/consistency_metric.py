from itertools import product
from typing import Any

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

    n_x = n_fh * L
    n_y = len(phi.keys())
    # print(f"length of a certain trade pattern x: n_fh({n_fh}) * L({L}) = {n_x}")
    # print(f"length of auxiliary variables y: {n_y}")

    # zero-value trades condition
    A_eq = np.zeros((n_fh, n_x + n_y), dtype=np.float32)
    for h, fh in fh_2_idx.keys():
        idx = fh_2_idx[(h, fh)]
        A_eq[idx][idx * L: (idx + 1) * L] = np.frombuffer(fh, dtype=np.float32)
    b_eq = np.zeros((n_fh,), dtype=np.float32)

    # print("------")
    # print("zero-value trade conditions:")
    # print(f"A_eq: {A_eq}")
    # print(f"b_eq: {b_eq}")
    # print("------")

    # theorem conditions
    n_f = len(phi.keys())
    n_inequalities = n_f * L
    A_ub = np.zeros((n_inequalities, n_x + n_y))
    for f_idx, f in enumerate(phi.keys()):
        f = np.frombuffer(f, dtype=np.float32).reshape(N_A, L)    # joint posterior
        for theta in range(L):    
            A_ub[f_idx * L + theta][n_x + f_idx] = -1    # -y(f)
            for h in range(N_A):
                f_h = f[h]
                x_idx = fh_2_idx[(h, f_h.tobytes())]
                A_ub[f_idx * L + theta][x_idx * L + theta] = 1
    b_ub = np.zeros((n_inequalities,), dtype=np.float32)

    # print("------")
    # print("theorem conditions:")
    # print(f"number of all possible f: {n_f}")
    # print(f"number of ineuqalities: {n_inequalities}")
    # print(f"A_ub: {A_ub}")
    # print(f"b_ub: {b_ub}")
    # print("------")

    # optimization target
    c = np.zeros((n_x + n_y,), dtype=np.float32)
    for f_idx, f in enumerate(phi.keys()):
        c[n_x + f_idx] = phi[f]
    # here x and y should be unbounded
    # but for normalization, bound x between (-1,1)
    bounds = [(-1, 1) for _ in range(n_x)] + [(None, None) for _ in range(n_y)]

    # print("------")
    # print("final optimize conditions")
    # print(f"A_eq.shape: {A_eq.shape}")
    # print(f"b_eq.shape: {b_eq.shape}")
    # print(f"A_ub.shape: {A_ub.shape}")
    # print(f"b_ub.shape: {b_ub.shape}")
    # print(f"c: {c}")
    # print("------")

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    consistency_metric = res.fun

    # print(f"consistency_metric: {consistency_metric}")
    # print(f"decision variables: {res.x}")
    # print(f"slack: {res.slack}")
    # print(f"con: {res.con}")
    # print(f"fun: {res.fun}")

    assert consistency_metric is not None, "linear programming returns None"

    return consistency_metric


if __name__=="__main__":
    label_nums = 2
    N_A = 2

    f1 = np.array([[0.9, 0.1], [0.5, 0.5]], dtype=np.float32)
    f2 = np.array([[0.1, 0.9], [0.5, 0.5]], dtype=np.float32)
    phi1 = phi2 = 0.5
    phi = {}
    phi[f1.tobytes()] = phi1
    phi[f2.tobytes()] = phi2

    print("------")
    print("joint posterior beliefs distribution: phi")
    for k in phi.keys():
        print(f"f: {np.frombuffer(k, dtype=np.float32).reshape(N_A, label_nums)}")
        print(f"phi(f): {phi[k]}")
    print("------")

    consistency_metric = calculate_consistency_metric(phi, N_A, label_nums)
    print(f"consistency metric: {consistency_metric}")


    '''
    print("-------")

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
