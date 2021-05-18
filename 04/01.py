#%%
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
#%%
w_method=gp.Model("VRS_model")
vrs_v[r,i]=w_method.addVar(vtype=gp.GRB.CONTINUOUS,name="%s"%(INPUTS[i]))
w_method.update()
w_method.setObjective(gp.quicksum(vrs_u[r,j]*Y[r][j] for j in range(O))-u0[r],gp.GRB.MAXIMIZE)
w_method.addConstr(gp.quicksum(vrs_v[r,i]*X[r][i] for i in range(I))==1)
w_method.optimize()
record[r] = np.round([v.x for v in w_method.getVars()] + [w_method.objVal] + [crs_model.objVal], 6)
