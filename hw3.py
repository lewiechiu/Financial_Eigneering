import numpy as np
import random

import scipy as sp
from scipy.stats import norm

K = 30
r = 0.09
T = 0.3
n = 5
S0s = [35, 39, 31, 20, 25]
qs = [0.01, 0.03, 0.02, 0, 0.04]
sigmas = [0.5, 0.2, 0.3, 0.2, 0.4]
rhos = np.random.rand(n, n)
Zs = np.random.normal(size = (n, 10000))


for i in range(n):
    for j in range(i, n):
        if i == j:
            rhos[i, j] = 1
        else:
            rhos[i, j] = ((np.random.rand() * 2) - 1) * 0.5
for i in range(n):
    for j in range(i):
        rhos[i, j] = rhos[j, i]
# print(rhos)

cnt = 0
while True:
    # Calculating the Covariance matrix
    C = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            C[i, j] = sigmas[i] * sigmas[j] * rhos[i, j]
    C *= T
    cnt +=1 
    print(cnt)
    try:
        np.linalg.cholesky(C)
        break
    except:
        continue
    
# print(rhos)
mus = [0] * n
for i in range(n):
    mus[i] = np.log(S0s[i]) + (r - qs[i] + (sigmas[i]**2 / 2)) * T

A = np.zeros((n, n))
print("Covariance")
print(C)

def Cholesky():
    global C
    global A
    # print(C)
    A[0, 0] = np.sqrt(C[0,0])
    for i in range(1, n):
        A[0, i] = C[0, i] / A[0, 0]
    for i in range(1, n - 1): 
        A[i ,i] = np.sqrt(C[i, i] - np.sum([A[k, i] ** 2 for k in range(i -1)]))
        
        for j in range(i+1, n):
            A[i, j] = 1 / A[i, i] * (C[i, j] - np.sum([A[k, i] * A[k, j] for k in range(i)]))
    A[n - 1, n - 1] = np.sqrt(C[n-1, n-1] - np.sum([A[k, n-1] ** 2 for k in range(n-1)]))
        
Cholesky()

print("Compare with Above Cov matrix")
print(np.matmul(A.T, A))
Zs_ = np.matmul(Zs.T, A)

