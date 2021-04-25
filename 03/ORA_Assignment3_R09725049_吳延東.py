#%%
from pulp import *
import numpy as np
import scipy.stats
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
    p = LpVariable(name="p", lowBound=0, upBound=None, cat="continuous")
    q = LpVariable(name="q", lowBound=0, upBound=None, cat="continuous")
    xp = []
    xp.append(LpVariable.dicts(name="x_p0", indexs=N_INDEX, lowBound=0, upBound=None, cat="continuous"))
    xp.append(LpVariable.dicts(name="x_p1", indexs=N_INDEX, lowBound=0, upBound=None, cat="continuous"))
    xq = []
    xq.append(LpVariable.dicts(name="x_q0", indexs=N_INDEX, lowBound=0, upBound=None, cat="continuous"))
    xq.append(LpVariable.dicts(name="x_q1", indexs=N_INDEX, lowBound=0, upBound=None, cat="continuous"))
    p0 = LpVariable.dicts(name="p0", indexs=N_INDEX, lowBound=0, upBound=None, cat="continuous")
    p1 = LpVariable.dicts(name="p1", indexs=N_INDEX, lowBound=0, upBound=None, cat="continuous")
    q0 = LpVariable.dicts(name="q0", indexs=N_INDEX, lowBound=0, upBound=None, cat="continuous")
    q1 = LpVariable.dicts(name="q1", indexs=N_INDEX, lowBound=0, upBound=None, cat="continuous")
    # binary = LpVariable(name="bi", lowBound=0, upBound=1, cat="binary")
    xj = LpVariable.dicts(name="xj", indexs=N_INDEX, lowBound=0, upBound=None, cat="continuous")
    xk = LpVariable.dicts(name="xk", indexs=N_INDEX, lowBound=0, upBound=None, cat="continuous")
    ra0 = LpVariable.dicts(name="ra0", indexs=M_INDEX, lowBound=0, upBound=None, cat="continuous")
    ra1 = LpVariable.dicts(name="ra1", indexs=M_INDEX, lowBound=0, upBound=None, cat="continuous")
    sa0 = LpVariable.dicts(name="sa0", indexs=M_INDEX, lowBound=0, upBound=None, cat="continuous")
    sa1 = LpVariable.dicts(name="sa1", indexs=M_INDEX, lowBound=0, upBound=None, cat="continuous")

    ## objective function
    prob += lpSum(
        # [V[0][i] * (p0[i] + q0[i]) for i in N_INDEX]
        # 6 * (p + q)
        [V[j][k] * (xp[j][k] + xq[j][k]) for j in N_INDEX for k in N_INDEX if j < k]
        # ) + lpSum(
        # [V[1][i] * (p0[i] + q0[i]) for i in N_INDEX]

        ) + lpSum(
        [W[0][i] * (ra0[i] + sa0[i]) for i in M_INDEX]
        
        ) + lpSum(
        [W[1][i] * (ra1[i] + sa1[i]) for i in M_INDEX] 
        )

    ## constraints
    # prob += xj[0] - q + p == xj[1]
    # prob += p0[0] == 0
    # prob += q0[1] == 0
    # prob += xj[0] - q0[1] + p0[0] == xj[1]
    for j in N_INDEX:
        for k in N_INDEX:
            if j < k:
                prob += xj[j] - xq[j][k] + xp[j][k] == xj[k]
    # prob += p - q >= 0.00001 + 10000 * binary
    # prob += p - q <= 0.00001 + 10000 * (1-binary)
    for i in M_INDEX:
        for j in N_INDEX:
            prob += a[i] == xj[j] - ra0[i] + sa0[i]
    for i in M_INDEX:
        for j in N_INDEX:
            prob += a[i] == xj[j] - ra1[i] + sa1[i]
    ## solve
    prob.solve()
    return prob
#%%
def prob_result(prob):
    print("Status for %s:" %prob.name, LpStatus[prob.status])

    #印出解及目標值
    for v in prob.variables():
        print(v.name, "=", v.varValue)
    print('obj=',value(prob.objective))
    pass
## 1a
prob_x = prob_sep(prob_name="1a_x", axis=0)
prob_result(prob=prob_x)
print("\n\n------------\n\n")
prob_y = prob_sep(prob_name="1a_y", axis=1)
prob_result(prob=prob_y)

#%%

#%%

#%%

#%%
