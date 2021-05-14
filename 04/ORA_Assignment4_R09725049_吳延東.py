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
DMU = [dep for s in SCHOOLS for dep in SCHOOL_DATA[s]["Department"]]

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
# I=2
# O=3
# INPUTS = ["zz", "xx"]
# OUTPUTS = ["cc", "vv", "bb"]
# # X、Y為各DMU的投入與產出    
# DMU = ["A", "B", "C", "D", "E"]
# X = {'A': [11, 14], 'B': [7, 7], 'C': [11, 14], 'D': [14, 14], 'E': [14, 15]}
# Y = {
#     'A': [2, 2, 1],
#     'B': [1, 1, 1],
#     'C': [1, 1, 2],
#     'D': [2, 3, 1],
#     'E': [3, 2, 3]
#     }

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
    
    record[r] = np.round([v.x for v in vrs_model.getVars()] + [vrs_model.objVal] + [crs_model.objVal], 3)


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
col = [v.varName for v in vrs_model.getVars()] + ["TE", "SE"]
result = pd.DataFrame(data=record).T
result.columns = col
result["OE"] = np.round(result["SE"] / result["TE"], 3)
result["return to scale"] = [judge_RS(u0=u0) for u0 in result["u_0"]]
result
#%%
result.to_csv("DEA_result.csv")
#%%
