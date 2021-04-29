#%%
from pulp import *
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
#%%

def solve_prob(LpProb):
    """[summary]

    Args:
        LpProb (LpProblem): [description]
    """
    LpProb.solve()
    #查看目前解的狀態
    print("Status:", LpStatus[LpProb.status])

    #印出解及目標值
    for v in LpProb.variables():
        print(v.name, "=", v.varValue)
    print('obj=',value(LpProb.objective))
    pass
#%%
## 1
### (a)
### (b)
### (c)
## add parameters
W = [
    [5, 3, 0, 0], 
    [0, 1, 8, 4]
    ]

V = [
    [0, 6],
    [6, 0]
    ]

A = [0, 4, 6, 10]
B = [2, 0, 8, 4]

M_INDEX = [0, 1, 2, 3]
N_INDEX = [0, 1]

#%%
def prob_sep(prob_name="problem_s", axis=0):
    if axis:
        a = B
    else:
        a = A
    ## problem
    prob = LpProblem("%s" %prob_name, LpMinimize)

    ## add variables
    p = []
    p.append(LpVariable.dicts(name="p0", indexs=N_INDEX, lowBound=0, upBound=None, cat="continuous"))
    p.append(LpVariable.dicts(name="p1", indexs=N_INDEX, lowBound=0, upBound=None, cat="continuous"))
    q = []
    q.append(LpVariable.dicts(name="q0", indexs=N_INDEX, lowBound=0, upBound=None, cat="continuous"))
    q.append(LpVariable.dicts(name="q1", indexs=N_INDEX, lowBound=0, upBound=None, cat="continuous"))
    x = LpVariable.dicts(name="x", indexs=N_INDEX, lowBound=0, upBound=None, cat="continuous")
    r = []
    r.append(LpVariable.dicts(name="r0", indexs=M_INDEX, lowBound=0, upBound=None, cat="continuous"))
    r.append(LpVariable.dicts(name="r1", indexs=M_INDEX, lowBound=0, upBound=None, cat="continuous"))
    s = []
    s.append(LpVariable.dicts(name="s0", indexs=M_INDEX, lowBound=0, upBound=None, cat="continuous"))
    s.append(LpVariable.dicts(name="s1", indexs=M_INDEX, lowBound=0, upBound=None, cat="continuous"))


    ## objective function
    prob += lpSum(
        [V[j][k] * (p[j][k] + q[j][k]) for j in N_INDEX for k in N_INDEX if j < k]
        ) + lpSum(
        [W[j][i] * (r[j][i] + s[j][i]) for i in M_INDEX for j in N_INDEX]
        )

    ## constraints
    for j in N_INDEX:
        for k in N_INDEX:
            if j < k:
                prob += x[j] - q[j][k] + p[j][k] == x[k]
    for i in M_INDEX:
        for j in N_INDEX:
            prob += a[i] == x[j] - r[j][i] + s[j][i]

    ## solve
    prob.solve()
    return prob
#%%
def prob_result(prob):
    print("Status for %s:" %prob.name, LpStatus[prob.status])
    target = []
    #印出解及目標值
    for v in prob.variables():
        print(v.name, "=", v.varValue)
        if "x" in v.name:
            target.append(v.varValue)
    print("---")
    print('obj=',value(prob.objective))
    return target
#%%
## 1b
prob_x = prob_sep(prob_name="1b_x", axis=0)
xs = prob_result(prob=prob_x)
print("\n\n------------\n\n")
prob_y = prob_sep(prob_name="1b_y", axis=1)
ys = prob_result(prob=prob_y)

#%%
## 1c
plt.figure(figsize=(8, 6))
plt.scatter(A, B)
plt.scatter(xs, ys, c="r")
plt.show()
print("藍色的點是舊的據點，紅色是新的")
#%%
import gurobipy as gp
from gurobipy import GRB
import math
#%%
# P: 長
# Q: 寬
P = [8] * 4 + [9] * 5 + [18] * 3
Q = [16] * 4 + [9] * 5 + [3] * 3
X_BAR = 18 * 3 + 9 * 5 + 16 * 4
Y_BAR = X_BAR
VAR = [0, 1]

BAR = [X_BAR, Y_BAR]

J = len(P)
INDEX_J = np.arange(J)
M = 10
INDEX_M = np.arange(M)
#%%
a = list()
ln_a = list()
for _ in np.linspace(3, X_BAR, num=M):
    a.append(_)
    ln_a.append(math.log(_))

b = list()
ln_b = list()
for _ in np.linspace(3, Y_BAR, num=M):
    b.append(_)
    ln_b.append(math.log(_))

t_X = list()
for j in range(0, M-1):
    t_X.append((ln_a[j+1] - ln_a[j]) / (a[j+1] - a[j]))

t_Y = list()
for j in range(0, M-1):
    t_Y.append((ln_b[j+1] - ln_b[j]) / (b[j+1] - b[j]))

plt.scatter(a, ln_a)
#%%
A = [a, b]
ln_A = [ln_a, ln_b]
SLOPE = [t_X, t_Y]
#%%
## problem
prob_3 = LpProblem("Cutting_Stock", LpMinimize)

## add variables
x = []
x.append(LpVariable(name="x", lowBound=0, upBound=BAR, cat="continuous"))
x.append(LpVariable(name="y", lowBound=0, upBound=BAR, cat="continuous"))

prime = []
prime.append(LpVariable.dicts(name="x_prime", indexs=INDEX_J, lowBound=0, upBound=BAR, cat="continuous"))
prime.append(LpVariable.dicts(name="y_prime", indexs=INDEX_J, lowBound=0, upBound=BAR, cat="continuous"))
s = LpVariable.dicts(name="s", indexs=INDEX_J, cat="binary")
u = (LpVariable.dicts(name="u", indexs=((i, j) for i in range(J) for j in range(J)), cat="binary"))
v = (LpVariable.dicts(name="j", indexs=((i, j) for i in range(J) for j in range(J)), cat="binary"))

w = []
w.append(LpVariable.dicts(name="w_x", indexs=INDEX_M, lowBound=0, upBound=float('inf'), cat="continuous"))
w.append(LpVariable.dicts(name="w_y", indexs=INDEX_M, lowBound=0, upBound=float('inf'), cat="continuous"))
r = []
r.append(LpVariable.dicts(name="r_x", indexs=INDEX_M, cat="binary"))
r.append(LpVariable.dicts(name="r_y", indexs=INDEX_M, cat="binary"))

## objective function
prob_3 += lpSum([
        ln_A[var][0] + 
        SLOPE[var][0] * (x[var] - A[var][0]) + 
        lpSum([
            (SLOPE[var][j] - SLOPE[var][j-1]) * 
            (r[var][j] * A[var][j] + x[var] - A[var][j] - w[var][j]) 
            for j in range(1, M-1)
            ]) 
    for var in VAR
    ])
## constraints
for var in VAR:
    for j in range(1, M):
        prob_3 += -A[var][-1] * r[var][j] <= x[var] - A[var][j]
        prob_3 += x[var] - A[var][j] <= A[var][-1] * (1 - r[var][j])

        prob_3 += -A[var][-1] * r[var][j] <= w[var][j]
        prob_3 += A[var][-1] * r[var][j] >= w[var][j]

        prob_3 += A[var][-1] * (r[var][j] - 1) + x[var] <= w[var][j]
        prob_3 += w[var][j] <= A[var][-1] * (1 - r[var][j]) + x[var]

        prob_3 += r[var][j] >= r[var][j-1]

    for i in INDEX_J:
        for k in INDEX_J:
            temp_v = v[i, k]
            if var == 1:
                temp_v = 1 - v[i, k]
            prob_3 += prime[var][i] - prime[var][k] + u[i, k] * BAR[var] + temp_v * BAR[var] >= 1/2 * (
                P[i] * s[i] + Q[i] * (1 - s[i]) + P[k] * s[k] + Q[k] * (1 - s[k]))

            prob_3 += prime[var][k] - prime[var][i] + (1 - u[i, k]) * BAR[var] + temp_v * BAR[var] >= 1/2 * (
                P[i] * s[i] + Q[i] * (1 - s[i]) + P[k] * s[k] + Q[k] * (1 - s[k]))
    prob_3 += x[var] <= BAR[var]

for i in INDEX_J:
    prob_3 += x[0] >= prime[0][i] + 1/2 *(P[i] * s[i] + Q[i] * (1 - s[i]))
    prob_3 += x[1] >= prime[1][i] + 1/2 *(P[i] * (1 - s[i]) + Q[i] * (s[i]))
    prob_3 += prime[0][i] - 1/2 *(P[i] * s[i] + Q[i] * (1 - s[i])) >= 0
    prob_3 += prime[1][i] - 1/2 *(P[i] * (1 - s[i]) + Q[i] * (s[i])) >= 0
#%%
## solve
prob_3.solve()
#%%
