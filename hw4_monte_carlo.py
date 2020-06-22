import numpy as np
import random
import scipy as sp
from scipy.stats import norm
from decimal import *
import math
import time

from hw4_config import *


delta_t = (T - t) / n
n_sim = 10000
n_rep = 20

now = time.time()
final_result = []
for _ in range(n_rep):
    result = []
    for exp_ in range(n_sim):
        max_St = Smax_t
        St_ = St
        for i in range(n):
            St_ = np.random.normal(np.log(St_) + (r - q - (sigma**2) / 2) * delta_t, sigma * np.sqrt(delta_t)) 
            St_ = np.exp(St_)
            max_St = max(max_St, St_)
        result.append(max(max_St - St_ , 0) * np.exp(-r * (T - t)))
    print(np.mean(result))
    final_result.append(np.mean(result))

print(np.mean(final_result) - 2 * np.std(final_result), np.mean(final_result) + 2 * np.std(final_result))

