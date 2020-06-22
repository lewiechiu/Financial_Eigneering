
import numpy as np
import time


from hw5_config import *

u = np.exp(sigma * np.sqrt(delta_t))
d = 1 / u
p = (np.exp((r - q) * delta_t) - d) / (u - d)

def isClose(x, y, tol = 1e-6):
    if abs(x - y) < tol:
        return True
    return False

def calc_Amax(i, j, passing_time = 0):
    ans = S_t * (passing_time + 1) + S_t * u * ((1 - u ** (i - j)) / (1 - u) ) + S_t * u ** (i - j) * d * ((1 - d ** j)/(1 - d))
    ans /= (i + (passing_time + 1))
    return ans

def calc_Amin(i, j, passing_time = 0):
    ans = S_t * (passing_time + 1) + S_t * d * (1 - d ** j) / (1 - d) + S_t * d ** j * u * ((1 - u ** (i - j)) / (1 - u) )
    ans /= (i + (passing_time + 1))
    return ans

def linear_interpolation(l, r, k):
    ans = 0
    ans = (M - k) / M * l + k / M * r
    return ans

def logarithmic_interpolation(l, r, k):
    ans = 0
    ans = np.exp((M - k) / M * np.log(l) + k / M * np.log(r))
    return ans

def linear_search_call_price(source, target):
    pos = 0
    for i in range(len(source)):
        if float(target) > source[i][0]:
            pos = i
            break
    if (source[pos - 1][0] - source[pos][0]) == 0:
        return source[pos][1]

    Wu = (source[pos - 1][0] - target) / (source[pos - 1][0] - source[pos][0])
    call = Wu * source[pos][1] + (1 - Wu) * source[pos - 1][1]
    return call

def binary_search_call_price(source, target):
    target = round(target, 8)
    l = 0
    r = len(source) - 1
    m = 0
    pos = -1
    # 4, 3, ,2, 1, 0
    while r >= l: 
        m = (l + r) // 2
        if source[m, 0] > target: 
            l = m + 1
        elif source[m, 0] < target: 
            r = m - 1
        else: 
            pos = m
            break
    if pos == -1:
        pos = l
    if pos == len(source):
        pos = len(source) - 1
    # for i in range(len(source)):
    #     if float(target) > source[i][0]:
    #         p_ = i

    #         if p_ != pos :
    #             print(target, p_, source[p_][0], pos, source[pos][0])
    #         break
    # print("#### end")
    if (source[pos - 1][0] - source[pos][0]) == 0:
        return source[pos][1]
    Wu = (source[pos - 1][0] - target) / (source[pos - 1][0] - source[pos][0])
    call = Wu * source[pos][1] + (1 - Wu) * source[pos - 1][1]
    # find target in source's position
    return call

def linear_interp_call_price(source, target):
    A = round(source[0][0] - target, 8)
    B = round(source[0][0] - source[-1][0], 8)

    if B == 0:
        return source[0][1]
    if A / B > 1:
        print(A, B,source[0][0],source[-1][0], target)
    pos = int(np.floor(M * (A / B)))
    if pos == M:
        pos = M - 1
    # adjust pos
    for i in range(pos, len(source)):
        if float(target) > source[i][0]:
            pos = i
            break
    Wu = (source[pos - 1][0] - target) / (source[pos - 1][0] - source[pos][0])
    call = Wu * source[pos][1] + (1 - Wu) * source[pos - 1][1]
    # find target in source's position
    return call
# Generate the tree

tree = np.zeros((n+1, n+1, M+1, 2))

start = time.time()
print("Counter started:", start)

for row in range(n+1):

    Amax_ij = calc_Amax(n, row, passing_time)
    Amin_ij = calc_Amin(n, row, passing_time)
    for node_ in range(M + 1):
        A_njk = 0
        if linearly_spaced :
            A_njk = linear_interpolation(Amax_ij, Amin_ij, node_)
        else:
            A_njk = logarithmic_interpolation(Amax_ij, Amin_ij, node_)
        tree[row][n][node_] = [A_njk , max(A_njk - K, 0)]
        # tree[row][n][node_] = linear_interpolation(Amax_ij, Amin_ij, node_)

# Backward induction
print(passing_time)
for col in range(n - 1, -1, -1):
    for row in range(col + 1):
        for k in range(M+1):
            Amax_ij = calc_Amax(col, row, passing_time)
            Amin_ij = calc_Amin(col, row, passing_time)
            if linearly_spaced :
                tree[row][col][k][0] = linear_interpolation(Amax_ij, Amin_ij, k)
            else:
                tree[row][col][k][0] = logarithmic_interpolation(Amax_ij, Amin_ij, k)
            Au = (col + passing_time + 1) * tree[row][col][k][0] + S_t * u ** (col + 1 - row) * d ** row
            Au /= (col + passing_time + 2)
            Ad = (col + passing_time + 1) * tree[row][col][k][0] + S_t * u ** (col + 1 - row - 1) * d ** (row + 1)
            Ad /= (col + passing_time + 2)
            if linear_search:
                Cu = linear_search_call_price(tree[row][col + 1][:], Au)
                Cd = linear_search_call_price(tree[row + 1][col + 1][:], Ad)
            elif binary_search:
                Cu = binary_search_call_price(tree[row][col + 1][:], Au)
                Cd = binary_search_call_price(tree[row + 1][col + 1][:], Ad)
            else:
                Cu = linear_interp_call_price(tree[row][col + 1][:], Au)
                Cd = linear_interp_call_price(tree[row + 1][col + 1][:], Ad)
            if european_option:
                tree[row][col][k][1] = (p * Cu + (1 - p) * Cd) * np.exp(-r * delta_t)
            else:
                tree[row][col][k][1] = max(tree[row][col][k][0] - K, (p * Cu + (1 - p) * Cd) * np.exp(-r * delta_t))
            
    #     break
    # break

print(tree[0][0][:][1])

print("finished",time.time() - start)
# 5.058046234270835 5.585434173133173