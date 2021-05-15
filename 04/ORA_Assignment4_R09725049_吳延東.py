#%%
from pandas.core.indexes import datetimes
# from pulp import *
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import gurobipy as gp
import pandas as pd
#%%

data = pd.read_excel("ORA_Assignment_04_DEA.xlsx", header=0)
#%%
SCHOOLS = [item for item in data["School"] if isinstance(item, str)]
#%%
SCHOOL_DATA = {}
for index, row in data.iterrows():
    if isinstance(row["School"], str):
        school = row["School"]
        SCHOOL_DATA[school] = pd.DataFrame(data=None, columns=data.columns[1:])
    else:
        SCHOOL_DATA[school] = SCHOOL_DATA[school].append(pd.DataFrame(pd.DataFrame(data=row[1:]).T, columns=data.columns[1:]))
#%%
I = 3
O = 3
INPUTS = [i for i in  data.columns[2:2+I]]
OUTPUTS = [o for o in  data.columns[2+I:2+I+O]]
#%%
for index, row in SCHOOL_DATA[SCHOOLS[0]].iterrows():
    print([row[i] for i in INPUTS])
    print()

#%%
DMU = []
X = {}
Y = {}
for s in SCHOOLS:
    for index, row in SCHOOL_DATA[s].iterrows():
        DMU = DMU + [row.iloc[0]]
        X[row.iloc[0]] = [i for i in row.iloc[1:4]]
        Y[row.iloc[0]] = [o for o in row.iloc[4:]]

#%%
record = {}  

## cal technical efficiency (VRS) and scale efficiency (CRS) for each DMU
for r in DMU:            
    
    ## VRS
    vrs_model=gp.Model("VRS_model")
    
    ## add variables
    vrs_v, vrs_u,u0={},{},{}
    for i in range(I):
        vrs_v[r,i]=vrs_model.addVar(vtype=gp.GRB.CONTINUOUS,name="%s"%(INPUTS[i]))
    
    for j in range(O):
        vrs_u[r,j]=vrs_model.addVar(vtype=gp.GRB.CONTINUOUS, name="%s"%(OUTPUTS[j]))
    u0[r]=vrs_model.addVar(lb=-1000,vtype=gp.GRB.CONTINUOUS, name="u_0")
    
    vrs_model.update()
    
    ## objective
    vrs_model.setObjective(gp.quicksum(vrs_u[r,j]*Y[r][j] for j in range(O))-u0[r],gp.GRB.MAXIMIZE)
    
    ## add constraints
    vrs_model.addConstr(gp.quicksum(vrs_v[r,i]*X[r][i] for i in range(I))==1)
    for k in DMU:
        vrs_model.addConstr(gp.quicksum(vrs_u[r,j]*Y[k][j] for j in range(O))-gp.quicksum(vrs_v[r,i]*X[k][i] for i in range(I))-u0[r] <=0)
    
    ## solve
    vrs_model.optimize()
    
    ## cal scale efficiency (CRS) for each dmu    
    crs_model=gp.Model("CRS_model")
    
    ## add variables
    crs_v, crs_u = {},{}
    for i in range(I):
        crs_v[r,i]=crs_model.addVar(vtype=gp.GRB.CONTINUOUS,name="v_%s%d"%(r,i))
    
    for j in range(O):
        crs_u[r,j]=crs_model.addVar(vtype=gp.GRB.CONTINUOUS,name="u_%s%d"%(r,j))
    
    crs_model.update()
    
    ## objective
    crs_model.setObjective(gp.quicksum(crs_u[r,j] * Y[r][j] for j in range(O)), gp.GRB.MAXIMIZE)
    
    ## add constraints
    crs_model.addConstr(gp.quicksum(crs_v[r,i] * X[r][i] for i in range(I)) == 1)
    for k in DMU:
        crs_model.addConstr(gp.quicksum(crs_u[r,j]*Y[k][j] for j in range(O)) - gp.quicksum(crs_v[r,i]*X[k][i] for i in range(I)) <= 0)
    
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
col = [v.varName for v in vrs_model.getVars()] + ["TE", "OE"]
result = pd.DataFrame(data=record).T
result.columns = col
result["SE"] = np.round(result["OE"] / result["TE"], 3)
result["return to scale"] = [judge_RS(u0=u0) for u0 in result["u_0"]]
result
#%%
result.to_csv("DEA_result.csv", float_format='%.6f')
#%%
result_read = pd.read_csv("DEA_result.csv")
#%%

## 3D plotting
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from pylab import figure
#%%
plot_data = pd.DataFrame(data=result_read["u_0"])
plot_data["TE"] = list(result_read["TE"])
plot_data["OE"] = list(result_read["OE"])

m = np.array(plot_data)# m is an array of (x,y,z) coordinate triplets

fig = figure(figsize=(8, 20))
ax = Axes3D(fig)

dot_name = list(plot_data.index)

for i in range(len(m)): #plot each point + it's index as text above
 ax.scatter(m[i,0],m[i,1],m[i,2], color='b') 
 ax.text(m[i,0],m[i,1],m[i,2], '%s' % (dot_name[i]), size=5, zorder=1, color='k') 

ax.set_zlabel('OE')
ax.set_ylabel('TE')
ax.set_xlabel('u_0')
pyplot.show()
#%%
