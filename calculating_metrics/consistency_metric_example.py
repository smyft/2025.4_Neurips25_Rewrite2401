from itertools import product
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import torch
import torchvision.transforms as transforms
from consistency_metric import calculate_consistency_metric
from scipy.optimize import linprog

"""
test the check theorem part
simulate some cases
"""

if __name__=="__main__":
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

        '''
        print("------")
        print("joint posterior beliefs distribution: phi")
        for k in phi.keys():
            print(f"f: {np.frombuffer(k, dtype=np.float32).reshape(N_A, label_nums)}")
            print(f"phi(f): {phi[k]}")
        print("------")
        '''

        metric = calculate_consistency_metric(phi, N_A, label_nums)
        metrics.append(metric)
        # print(f"consistency metric: {metric}")

    plt.figure()
    plt.plot(range_c, metrics)
    plt.xlabel('c')
    plt.ylabel('metric')
    plt.savefig('results/consistency_metric_example.png')

    '''

    tikz_code = tikzplotlib.get_tikz_code()
    with open('tikz_code.txt', 'w') as f:
        f.write(tikz_code)
    print(tikz_code)
    '''

    plt.show()    

    '''
    for c, metric in zip(range_c, metrics):
        print(f"c: {c}\tmetric:{metric}")
        if(abs(metric - (8/7*(c - 3/4))) > 1e-6):
            print("error")
    '''
    '''
    r = 3 / 4
    c = 1 / 4
    f1 = np.array([[1 - r, r], [1 - r, r]])
    f2 = np.array([[r, 1 - r], [r, 1 - r]])
    f3 = np.array([[1 - r, r], [r, 1 - r]])
    f4 = np.array([[r, 1 - r], [1 - r, r]])
    # phi1 = phi2 = (r ** 2 + (1 - r) ** 2) / 2
    # phi3 = phi4 = r * (1 - r)
    phi1 = phi2 = c / 2
    phi3 = phi4 = 1 / 2 - c / 2
    phi = {}
    phi[f1.tobytes()] = phi1
    phi[f2.tobytes()] = phi2
    phi[f3.tobytes()] = phi3
    phi[f4.tobytes()] = phi4
    N_A = 2
    label_nums = 2
    '''
