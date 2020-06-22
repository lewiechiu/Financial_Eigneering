import numpy as np
import random
import scipy as sp
from scipy.stats import norm
import math
import time

from hw5_config import *


delta_t = (T - t) / n
n_sim = 30000
n_rep = 20

# now = time.time()
# final_result = []
# for _ in range(n_rep):
#     result = []
#     for exp_ in range(n_sim):
#         avg_St = S_ave_t
#         St_ = S_t
#         for i in range(n):
#             St_ = np.random.normal(np.log(St_) + (r - q - (sigma**2) / 2) * delta_t, sigma * np.sqrt(delta_t)) 
#             St_ = np.exp(St_)
#             avg_St += St_
#         avg_St /= n
#         result.append(max(avg_St - K , 0) * np.exp(-r * (T - t)))
#     print(np.mean(result))
#     final_result.append(np.mean(result))

# print(np.mean(final_result) - 2 * np.std(final_result), np.mean(final_result) + 2 * np.std(final_result))


now = time.time()
final_result = []
for _ in range(n_rep):

    avg_St = [S_ave_t * passing_time for i in range(n_sim)]
    St_ = [S_t  for i in range(n_sim)]
    for i in range(n):
        St_ = np.random.normal(np.log(St_) + (r - q - (sigma**2) / 2) * delta_t, sigma * np.sqrt(delta_t)) 
        St_ = np.exp(St_)
        avg_St += St_
    avg_St /= (n + passing_time)
    result = [max(mean - K , 0) * np.exp(-r * (T - t)) for mean in avg_St]
    print(np.mean(result))
    final_result.append(np.mean(result))

print(np.mean(final_result) - 2 * np.std(final_result), np.mean(final_result) + 2 * np.std(final_result))

