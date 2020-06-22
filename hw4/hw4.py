import numpy as np
import random
import scipy as sp
from scipy.stats import norm
from decimal import *
import math
from hw4_config import *

np.set_printoptions(suppress = True)


delta_t = (T - t) / n
n_sim = 10000
n_rep = 20


u = np.exp(sigma * np.sqrt(delta_t))
d = 1 / u
p = (np.exp((r - q) * delta_t) - d) / (u - d)
print("u: {} d: {} p: {}".format(u, d, p))


S = [[0.] * (n + 1)] * (n+1)
Smax = [[list() for i in range(n + 1)] for j in range(n+1)]
S = np.array(S)
# print("Smax:",Smax)

put_price = [[{} for i in range(n + 1)] for j in range(n+1)]
# print("Put price:", put_price)
possible_stock_price = [St * (u ** i) for i in range(n, 0, -1)]
possible_stock_price += [St * (d ** i) for i in range(n + 1)]

print("Possible stock count:", len(possible_stock_price))
#######
"""
x x x x x x
  x x x x x
    x x x x
      x x x
        x x
          x
"""

def isClose(x, y, tol = 1e-12):
    if abs(x - y) < tol:
        return True
    return False

# Insert values of Stocks
# Will be filling in (0, n - 1), (0, n - 2), (1, n - 1), (1, n-2), (2, n-1)...

S[0][n] = possible_stock_price[0]
cur_row = 0
for i in range(1, len(possible_stock_price), 2):
    S[cur_row][n-1] = possible_stock_price[i]
    S[cur_row + 1][n] = possible_stock_price[i+1]
    cur_row += 1

for col in range(n-2, -1, -1):
    for row in range(col+1):
        S[row][col] = S[row+1][col + 2]
print("S Price")
# print(S)

Smax[0][0] = [Smax_t]
for col in range(1, n+1):
    for row in range( col + 1):
        ## Check parent
        ## i, j  --> (i, j-1), (i-1, j-1)
        # print(row, col)
        if row - 1 >= 0 and col - 1 >= row:
            # Check both parent
            # print("Both parent")
            tmp = [S[row][col] if parent_max < S[row][col] else parent_max for parent_max in Smax[row - 1][col - 1]]
            tmp += [S[row][col] if parent_max < S[row][col] else parent_max for parent_max in Smax[row ][col - 1]]
            # print(S[row][col], set(tmp))
            Smax[row][col] = sorted(list(set(tmp)), reverse = True)
        elif row == col:
            # print("top left parent")
            tmp = [S[row][col] if parent_max < S[row][col] else parent_max for parent_max in Smax[row - 1][col - 1]]
            # print(S[row][col], set(tmp))
            Smax[row][col] = sorted(list(set(tmp)), reverse = True)
        else:
            # print("left parent")
            tmp = [S[row][col] if parent_max < S[row][col] else parent_max for parent_max in Smax[row ][col - 1]]
            # print(S[row][col], set(tmp))
            Smax[row][col] = sorted(list(set(tmp)), reverse = True)
print("S max:")
# print(Smax)
# print(Smax)

# Update the n-1 th column to 1 column
    # for each cell i, j:
        # update current put price if 
for col in range(n, -1, -1):
    for row in range(0, col + 1):
        # print(row, col)
        if col == n:
            put_price[row][col] = [(p_smax, p_smax - S[row][col])if p_smax - S[row][col] > 0 else (p_smax, 0) for p_smax in Smax[row][col]]
            put_price[row][col] = sorted(put_price[row][col], key = lambda x: x[0], reverse = True)
        else:
            Smax_PutPrice = []
            k_up_ = 0
            k_down_ = 0
            for _s_max in Smax[row][col]:
                # search in put_price[row][col + 1]
                    # if _s_max in put_price i, j+1:
                        # retrieve such value (_s_max, future_put_price)
                    # else:
                        # retrieve price = _s_max * u, in put_price i, j+1:
                put_up = -1
                put_down = -1
                while k_up_ < len(put_price[row][col + 1]):
                    if isClose(put_price[row][col + 1][k_up_][0], _s_max):
                        
                        put_up = put_price[row][col + 1][k_up_][1]
                        break
                    if _s_max >= put_price[row][col + 1][k_up_][0]:
                        break
                    k_up_ += 1
                if put_up == -1:
                    k_up_ = 0
                    while k_up_ < len(put_price[row][col + 1]):
                        if isClose(put_price[row][col + 1][k_up_][0], S[row][col] * u):
                            
                            put_up = put_price[row][col + 1][k_up_][1]
                            break
                        k_up_ += 1
                while k_down_ < len(put_price[row + 1][col + 1]):
                    if isClose(put_price[row + 1][col + 1][k_down_][0], _s_max):
                        put_down = put_price[row + 1][col + 1][k_down_][1]
                        break
                    if _s_max >= put_price[row + 1][col + 1][k_down_][0]:
                        break
                    k_down_ += 1
                put_value = (put_up * p + put_down * (1-p) ) * np.exp(-r * delta_t)
                if put_value < _s_max - S[row][col] and not european_option:
                    # Early exercise
                    put_value = _s_max - S[row][col]
                Smax_PutPrice.append((_s_max,  put_value) )
            put_price[row][col] = Smax_PutPrice


print(
"""
##### [FINAL] #####
put value: {}
""".format(put_price[0][0][0][1]))

