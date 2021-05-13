#%%
from pulp import *
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from gurobipy import*
import gurobipy as gp
import pandas as pd
#%%

VAR, E={}, {}  
I=2
O=3
#X、Y為各DMU的投入與產出    
DMU,X,Y=gp.multidict({('A'):[[11,14],[2,2,1]],('B'):[[7,7],[1,1,1]],('C'):[[11,14],[1,1,2]],('D'):[[14,14],[2,3,1]],('E'):[[14,15],[3,2,3]]})
def solve_DEA_vrs():
    m=gp.Model("VRS_model")
    
    ## add variables
    v,u,u0={},{},{}
    for i in range(I):
        v[r,i]=m.addVar(vtype=gp.GRB.CONTINUOUS,name="v_%s%d"%(r,i))
    
    for j in range(O):
        u[r,j]=m.addVar(vtype=gp.GRB.CONTINUOUS,name="u_%s%d"%(r,j))
    u0[r]=m.addVar(lb=-1000,vtype=gp.GRB.CONTINUOUS,name="u_0%s"%r)
    
    m.update()
    
    ## objective
    m.setObjective(gp.quicksum(u[r,j]*Y[r][j] for j in range(O))-u0[r],gp.GRB.MAXIMIZE)
    
    ## add constraints
    m.addConstr(gp.quicksum(v[r,i]*X[r][i] for i in range(I))==1)
    for k in DMU:
        m.addConstr(gp.quicksum(u[r,j]*Y[k][j] for j in range(O))-gp.quicksum(v[r,i]*X[k][i] for i in range(I))-u0[r] <=0)
    
    ## solve
    m.optimize()
    return m
## cal efficiency for each dmu
for r in DMU:            
        
    m=gp.Model("VRS_model")
    
    ## add variables
    v,u,u0={},{},{}
    for i in range(I):
        v[r,i]=m.addVar(vtype=gp.GRB.CONTINUOUS,name="v_%s%d"%(r,i))
    
    for j in range(O):
        u[r,j]=m.addVar(vtype=gp.GRB.CONTINUOUS,name="u_%s%d"%(r,j))
    u0[r]=m.addVar(lb=-1000,vtype=gp.GRB.CONTINUOUS,name="u_0%s"%r)
    
    m.update()
    
    ## objective
    m.setObjective(gp.quicksum(u[r,j]*Y[r][j] for j in range(O))-u0[r],gp.GRB.MAXIMIZE)
    
    ## add constraints
    m.addConstr(gp.quicksum(v[r,i]*X[r][i] for i in range(I))==1)
    for k in DMU:
        m.addConstr(gp.quicksum(u[r,j]*Y[k][j] for j in range(O))-gp.quicksum(v[r,i]*X[k][i] for i in range(I))-u0[r] <=0)
    
    ## solve
    m.optimize()
    
    E[r]="The efficiency of DMU %s:%0.3f and \n %s= %0.3f"%(r, m.objVal, u0[r].varName, u0[r].X)
    VAR[r] = [v.x for v in m.getVars()]


#%%
for r in DMU:
    print (E[r])
#%%
