#%%
from pandas.core.indexes import datetimes
# from pulp import *
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import gurobipy as gp
import pandas as pd
#%%

#%%
I = 1
O = 1
INPUTS = ["input"]
OUTPUTS = ["output"]
#%%
#%%
DMU = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

#%%
X = {}
Y = {}
inputs = [1, 2, 4, 6, 9, 5, 4, 10, 8]
outputs = [1, 4, 6, 7, 8, 3, 1, 7, 4]

for i in range(len(DMU)):
    X[DMU[i]] = [inputs[i]]
    Y[DMU[i]] = [outputs[i]]

#%%
record = {}  

## cal technical efficiency (VRS) and scale efficiency (CRS) for each DMU
for r in DMU:            
    
    ## VRS
    vrs_model=gp.Model("VRS_model")
    
    ## add variables
    vrs_v, vrs_u,u0={},{},{}
    for i in range(I):
        vrs_v[r,i]=vrs_model.addVar(vtype=gp.GRB.CONTINUOUS, name="%s"%(INPUTS[i]))
    
    for j in range(O):
        vrs_u[r,j]=vrs_model.addVar(vtype=gp.GRB.CONTINUOUS, name="%s" % (OUTPUTS[j]))
    u0[r]=vrs_model.addVar(lb=-1000, vtype=gp.GRB.CONTINUOUS, name="u_0")
    
    vrs_model.update()
    
    ## objective
    vrs_model.setObjective(gp.quicksum(vrs_u[r,j] * Y[r][j] for j in range(O))-u0[r], gp.GRB.MAXIMIZE)
    
    ## add constraints
    vrs_model.addConstr(gp.quicksum(vrs_v[r,i] * X[r][i] for i in range(I)) == 1)
    for k in DMU:
        vrs_model.addConstr(gp.quicksum(vrs_u[r,j] * Y[k][j] for j in range(O)) - gp.quicksum(vrs_v[r,i] * X[k][i] for i in range(I)) - u0[r] <= 0)
    
    ## solve
    vrs_model.optimize()
    
    ## cal scale efficiency (CRS) for each dmu    
    crs_model=gp.Model("CRS_model")
    
    ## add variables
    crs_v, crs_u = {},{}
    for i in range(I):
        crs_v[r,i]=crs_model.addVar(vtype=gp.GRB.CONTINUOUS, name="v_%s%d"%(r,i))
    
    for j in range(O):
        crs_u[r,j]=crs_model.addVar(vtype=gp.GRB.CONTINUOUS, name="u_%s%d"%(r,j))
    
    crs_model.update()
    
    ## objective
    crs_model.setObjective(gp.quicksum(crs_u[r,j] * Y[r][j] for j in range(O)), gp.GRB.MAXIMIZE)
    
    ## add constraints
    crs_model.addConstr(gp.quicksum(crs_v[r,i] * X[r][i] for i in range(I)) == 1)
    for k in DMU:
        crs_model.addConstr(gp.quicksum(crs_u[r,j] * Y[k][j] for j in range(O)) - gp.quicksum(crs_v[r,i] * X[k][i] for i in range(I)) <= 0)
    
    ## solve
    crs_model.optimize()
    
    record[r] = np.round([v.x for v in vrs_model.getVars()] + [vrs_model.objVal] + [crs_model.objVal], 6)


#%%
def judge_RS(u0):
    if u0 > 0:
        return "DRS"
    elif u0 < 0:
        return "IRS"
    return "CRS"
#%%
# print(VAR)
#%%
col = [v.varName for v in vrs_model.getVars()] + ["TE(VRS)", "OE(CRS)"]
test_result = pd.DataFrame(data=record).T
test_result.columns = col
test_result["SE"] = np.round(test_result["OE(CRS)"] / test_result["TE(VRS)"], 3)
test_result["return to scale"] = [judge_RS(u0=u0) for u0 in test_result["u_0"]]
test_result
#%%
# result.to_csv("DEA_result.csv", float_format='%.6f')
#%%
# result_read = pd.read_csv("DEA_result.csv")

#%%
## 3D plotting
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from pylab import figure
#%%
plot_data = pd.DataFrame(data=test_result["TE(VRS)"])
plot_data["OE(CRS)"] = list(test_result["OE(CRS)"])
plot_data["u_0"] = list(test_result["u_0"])
#%%
m = np.array(plot_data)# m is an array of (x,y,z) coordinate triplets

fig = figure(figsize=(8, 8))
ax = Axes3D(fig)

dot_name = list(plot_data.index)

for i in range(len(m)): #plot each point + it's index as text above
 ax.scatter(m[i,0],m[i,1],m[i,2], color='b') 
 ax.text(m[i,0],m[i,1],m[i,2], '%s' % (dot_name[i]), size=20, zorder=1, color='k') 

ax.set_xlabel('TE(VRS)')
ax.set_ylabel('OE(CRS)')
ax.set_zlabel('u_0')
pyplot.show()
#%%