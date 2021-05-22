#%%
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
#%%
W = [2, 1]

w_method = gp.Model("W_method")
x1 = w_method.addVar(vtype=gp.GRB.CONTINUOUS,name="x1", lb=0)
x2 = w_method.addVar(vtype=gp.GRB.CONTINUOUS,name="x2", lb=0)
w_method.update()

w_method.setObjective(W[0]*(3*x1 + 5*x2) - W[1]*(2*x1 + 4*x2), gp.GRB.MAXIMIZE)

w_method.addConstr(x1 <= 4)
w_method.addConstr(2*x2 <= 12)
w_method.addConstr(3*x1 + 2*x2 <= 18)

w_method.optimize()
#%%
w_obj = w_method.objVal
#%%
w_x = w_method.x
w_profit = (3*w_x[0] + 5*w_x[1])
w_penalty = (2*w_x[0] + 4*w_x[1])
print("when profit : penalty is 2 : 1\n")
print("\tprofit is: %d" %w_profit)
print("\tpenalty is: %d" %w_penalty)
#%%
# epsion constraint method

f_panalty = gp.Model("f_panalty")

x1 = f_panalty.addVar(vtype=gp.GRB.CONTINUOUS,name="x1", lb=0)
x2 = f_panalty.addVar(vtype=gp.GRB.CONTINUOUS,name="x2", lb=0)
f_panalty.update()

f_panalty.setObjective(-(2*x1 + 4*x2), gp.GRB.MAXIMIZE)

f_panalty.addConstr(x1 <= 4)
f_panalty.addConstr(2*x2 <= 12)
f_panalty.addConstr(3*x1 + 2*x2 <= 18)

f_panalty.optimize()

f_panalty_min = gp.Model("f_panalty")

x1 = f_panalty_min.addVar(vtype=gp.GRB.CONTINUOUS,name="x1", lb=0)
x2 = f_panalty_min.addVar(vtype=gp.GRB.CONTINUOUS,name="x2", lb=0)
f_panalty_min.update()

f_panalty_min.setObjective(-(2*x1 + 4*x2), gp.GRB.MINIMIZE)

f_panalty_min.addConstr(x1 <= 4)
f_panalty_min.addConstr(2*x2 <= 12)
f_panalty_min.addConstr(3*x1 + 2*x2 <= 18)

f_panalty_min.optimize()
#%%
R = 3
T = np.arange(R)
c_methods = []
for t in T:
    c_methods.append(gp.Model("W_method"))
    e = f_panalty_min.objVal + (t / (R-1))*(f_panalty.objVal - f_panalty_min.objVal)
    x1 = c_methods[t].addVar(vtype=gp.GRB.CONTINUOUS,name="x1", lb=0)
    x2 = c_methods[t].addVar(vtype=gp.GRB.CONTINUOUS,name="x2", lb=0)
    c_methods[t].update()

    c_methods[t].setObjective(W[0]*(3*x1 + 5*x2) - W[1]*(2*x1 + 4*x2), gp.GRB.MAXIMIZE)

    c_methods[t].addConstr(-(2*x1 + 4*x2) >= e)
    c_methods[t].addConstr(x1 <= 4)
    c_methods[t].addConstr(2*x2 <= 12)
    c_methods[t].addConstr(3*x1 + 2*x2 <= 18)

    c_methods[t].optimize()
    print("\n\n===========\n\n")
#%%
print([c_methods[t].objVal for t in T])
print([c_methods[t].x for t in T])
#%%
print("when r = 3\n")
for t in T:
    c_profit = c_methods[t].objVal
    c_penalty = 2*c_methods[t].x[0] + 4*c_methods[t].x[1]
    print("\twhen t = %d" %t)
    print("\tprofit is: %d" %c_profit)
    print("\tpenalty is: %d" %c_penalty)
    print()
#%%
