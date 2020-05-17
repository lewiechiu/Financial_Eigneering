import numpy as np
import random

import scipy as sp
from scipy.stats import norm

n = 5
Zs = np.random.normal(size = (n, 10000))
C = np.cov(Zs)
A = np.matrix((n, n))

def Cholesky():
    global C
    global A
    A[0, 0] = np.sqrt(C[0,0])
    for i in range(1, n):
        A[0, i] = C[0, i] / A[0, 0]
    for i in range(n):
        A[i ,i] = np.sqrt(C[i, i] - np.sum([A[k, i] ** 2 for k in range(0, i)]))
        for j in range(n):
            A[i, j] = 

print(C)