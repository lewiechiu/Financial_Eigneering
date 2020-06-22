import numpy as np
import random

import scipy as sp
from scipy.stats import norm

K = 100
r = 0.1
T = 0.5
n = 5
S0s = np.array([95.0] * n)
qs = np.array([0.05] * n)
sigmas = np.array([0.5] * n)
total_experiment = 10000
rhos = np.random.rand(n, n)
n_exp = 20
mus = [np.log(S0s[i]) + (r - qs[i] - (sigmas[i]**2 / 2)) * T for i in range(n)]
A = np.zeros((n, n))

## Data preparation
for i in range(n):
    for j in range(i, n):
        if i == j:
            rhos[i, j] = 1.
        else:
            rhos[i, j] = 0.5
for i in range(n):
    for j in range(i):
        rhos[i, j] = rhos[j, i]
print(rhos)

## [GENERATE COV] Remove when given true value

cnt = 0
while True:
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = sigmas[i] * sigmas[j] * rhos[i, j]
    C *= T
    try:
        # print(np.linalg.cholesky(C))
        break
    except:
        continue


####################################

np.set_printoptions(suppress = True)
print("Covariance")
print(C)
def Cholesky(C, A):
    # print(C)
    A[0, 0] = np.sqrt(C[0,0])
    for i in range(1, n):
        A[0, i] = C[0, i] / A[0, 0]
    for i in range(1, n - 1):
        # print(i, np.sum([A[k, i] ** 2 for k in range(i )]))
        tmp = C[i, i] - np.sum([A[k, i] ** 2 for k in range(i )])
        if tmp < 0:
            tmp = 0
        A[i ,i] = np.sqrt(tmp)
        for j in range(i+1, n):
            A[i, j] = 1 / A[i, i] * (C[i, j] - np.sum([A[k, i] * A[k, j] for k in range(i)]))
    A[n - 1, n - 1] = np.sqrt(C[n-1, n-1] - np.sum([ A[k, n-1] ** 2 for k in range(n-1)]))
Cholesky(C, A)

# print(A)

## Calculation begins
############### Basic requirements
past_exp = []
print("Basic Requirements")
for num_exp in range(n_exp):
    Zs = np.random.normal(size = (total_experiment, n))
    # print(np.cov(Zs.T))
    Zs_ = np.matmul(Zs, A) # Zs_ is r
    # break
    for i in range(n):
        Zs_[:, i] = Zs_[:, i] + mus[i]
    Zs_ = np.exp(Zs_)
    # print(Zs_)
    # print(max(Zs_[i, :]))
    payoff = [max( max(Zs_[i, :])  - K, 0) for i in range(total_experiment)]

    # print(np.exp(-r * T) * np.mean(payoff))
    past_exp.append(np.exp(-r * T) * np.mean(payoff))

print(np.mean(past_exp))
print(np.mean(past_exp) - 2 * np.std(past_exp), np.mean(past_exp) + 2 * np.std(past_exp))

################ [Bonus 1]

# Z1_hat = [z_1 , -z_1]
# Moment match: scale s.d. to 1
past_exp = []
print("Bonus")
for num_exp in range(n_exp):
    Zs = np.random.normal(size = (int(total_experiment / 2), n))
    Zs = np.vstack((Zs, -Zs))
    Zs_ = np.zeros_like(Zs)
    # print(Zs.shape)

    for i in range(n):
        Zs[:, i] /= np.std(Zs[:, i])
    Zs_ = np.matmul(Zs, A) # Zs_ is r
    # print(np.cov(Zs_.T))
    for i in range(n):
        Zs_[:, i] = Zs_[:, i] + mus[i]
    Zs_ = np.exp(Zs_)
    payoff = [max( max(Zs_[i, :])  - K, 0) for i in range(total_experiment)]

    # print(np.exp(-r * T) * np.mean(payoff))
    past_exp.append(np.exp(-r * T) * np.mean(payoff))
    # break

print(np.mean(past_exp))
print(np.mean(past_exp) - 2 * np.std(past_exp), np.mean(past_exp) + 2 * np.std(past_exp))


# ## Bonus 2
print("Bonus 2")
# BB = np.cov(Zs_.T)
# B = np.zeros((n, n))
# Cholesky(BB, B)
# Zs_b2 = np.matmul(Zs_, np.linalg.inv(B))
# print("Covariance of converted Z_hat")
# print(np.cov(Zs_b2.T))

past_exp = []
for num_exp in range(n_exp):
    Zs = np.random.normal(size = (int(total_experiment / 2), n))
    Zs = np.vstack((Zs, -Zs))
    
    for i in range(n):
        Zs[:, i] /= np.std(Zs[:, i])
    BB = np.cov(Zs.T)
    B = np.zeros((n, n))
    Cholesky(BB, B)
    Zs_ = np.matmul(Zs, np.linalg.inv(B))
    Zs_ = np.matmul(Zs_, A) # Zs_ is r
    # print(np.cov(Zs_.T))
    for i in range(n):
        Zs_[:, i] = Zs_[:, i] + mus[i]
    Zs_ = np.exp(Zs_)
    payoff = [max( max(Zs_[i, :])  - K, 0) for i in range(total_experiment)]

    # print(np.exp(-r * T) * np.mean(payoff))
    past_exp.append(np.exp(-r * T) * np.mean(payoff))

print(np.mean(past_exp))
print(np.mean(past_exp) - 2 * np.std(past_exp), np.mean(past_exp) + 2 * np.std(past_exp))
