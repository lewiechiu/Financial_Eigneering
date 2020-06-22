import numpy as np
import random
import scipy as sp
from scipy.stats import norm
from decimal import *
import math
from hw4_config import *

np.set_printoptions(suppress = True)

n = 1000
n_sim = 10000
n_rep = 20
european_option = False
delta_t = (T - t) /  n
u = np.exp(sigma * np.sqrt(delta_t))
d = 1 / u
p = (np.exp((r - q) * delta_t) - d) / (u - d) # p rise
print("u: {} d: {} p: {}".format(u, d, p))

u_matrix = np.zeros((n+1, n+1))
for i in range(n+1):
    u_matrix[i, -1] = max(u ** i - 1, 0)

for col in range(n-1, -1, -1):
    for row in range(col + 1):
        if row - 1 < 0:
            # Can go up
            u_matrix[row, col] = p * u * u_matrix[row, col + 1] + (1 - p) * d* u_matrix[row + 1, col + 1]
        else:
            u_matrix[row, col] = p * u * u_matrix[row - 1, col + 1] + (1 - p) * d * u_matrix[row + 1, col + 1] 
        u_matrix[row, col] = u_matrix[row, col] * np.exp(-r * delta_t) 
        if european_option == False:
            u_matrix[row, col] = max(u_matrix[row, col], (u ** row) - 1)
price = u_matrix[0, 0] * St
# print(price)
# price *= np.exp(-r * (T))
# print(u_matrix[:, -1])
# print(u_matrix[0:2, 0:2])
print(
"""##### [FINAL] #####
put value: {}
""".format(price))