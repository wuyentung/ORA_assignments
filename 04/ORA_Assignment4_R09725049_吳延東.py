#%%
from pandas.core.indexes import datetimes
from pulp import *
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import gurobipy as gp
import pandas as pd
#%%

VAR, E={}, {}  
I=2
O=3
# X、Y為各DMU的投入與產出    

DMU,X,Y=gp.multidict(
        {
        ('A'):[[11,14],[2,2,1]], 
        ('B'):[[7,7],[1,1,1]], 
        ('C'):[[11,14],[1,1,2]], 
        ('D'):[[14,14],[2,3,1]], 
        ('E'):[[14,15],[3,2,3]]
        }
    )

## cal technical efficiency (VRS) and scale efficiency (CRS) for each DMU
for r in DMU:            
        
    vrs_model=gp.Model("VRS_model")
    
    ## add variables
    vrs_v, vrs_u,u0={},{},{}
    for i in range(I):
        vrs_v[r,i]=vrs_model.addVar(vtype=gp.GRB.CONTINUOUS,name="v_%s%d"%(r,i))
    
    for j in range(O):
        vrs_u[r,j]=vrs_model.addVar(vtype=gp.GRB.CONTINUOUS,name="u_%s%d"%(r,j))
    u0[r]=vrs_model.addVar(lb=-1000,vtype=gp.GRB.CONTINUOUS,name="u_0%s"%r)
    
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
    
    VAR[r] = np.round([v.x for v in vrs_model.getVars()] + [vrs_model.objVal] + [crs_model.objVal], 3)


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
d = pd.DataFrame(data=VAR).T
d.columns = col
d["OE"] = np.round(d["SE"] / d["TE"], 3)
d["return to scale"] = [judge_RS(u0=u0) for u0 in d["u_0E"]]
d
#%%
#%%
data = pd.read_excel("ORA_Assignment_04_DEA.xlsx", header=0)
#%%
schools = [item for item in data["School"] if isinstance(item, str)]
#%%
School_data = {}
for index, row in data.iterrows():
    # print(row[1:-1])
    if isinstance(row["School"], str):
        school = row["School"]
        School_data[school] = pd.DataFrame(data=None, columns=data.columns[1:-1])
    else:
        School_data[school] = School_data[school].append(pd.DataFrame(pd.DataFrame(data=row[1:-1]).T, columns=data.columns[1:-1]))
#%%
#%%
