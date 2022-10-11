# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
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
schools, deps = [], []
for index, row in data.iterrows():
    if isinstance(row["School"], str):
        school = row["School"]
        SCHOOL_DATA[school] = pd.DataFrame(data=None, columns=data.columns[1:])
    else:
        SCHOOL_DATA[school] = SCHOOL_DATA[school].append(pd.DataFrame(pd.DataFrame(data=row[1:]).T, columns=data.columns[1:]))
        schools.append(school)
        deps.append(row["Department"])
#%%
IDX = pd.MultiIndex.from_arrays([schools, deps], names=["School", "Department"])
collect = pd.concat([SCHOOL_DATA[school].iloc[:] for school in SCHOOLS]).drop(["Department"], axis=1)
VALUE_DATA = pd.DataFrame(data=np.array(collect), columns=collect.columns, index=IDX)
VALUE_DATA.head()


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
result = result.set_index(keys=IDX)
result


#%%
result.to_csv("DEA_result.csv", float_format='%.6f')


#%%
result_read = (pd.read_csv("DEA_result.csv", index_col=0))
result_read = pd.DataFrame(data=np.array(result_read), columns=result_read.columns, index=IDX)
result_read.head()


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

fig = figure(figsize=(10, 20))
ax = Axes3D(fig)

dot_name = deps

for i in range(len(m)): #plot each point + it's index as text above
 ax.scatter(m[i,0],m[i,1],m[i,2], color='b') 
 ax.text(m[i,0],m[i,1],m[i,2], '%s' % (dot_name[i]), size=5, zorder=1, color='k') 

ax.set_zlabel('OE')
ax.set_ylabel('TE')
ax.set_xlabel('u_0')
pyplot.show()


#%%
## find efficient point
best = pd.DataFrame(columns=result_read.columns)
for index, row in result_read.iterrows():
    if row["TE"] == 1:
        if row["OE"] == 1:
            best = best.append(row)
best

#%%
def get_value_df(df):
    df_copy = df.copy()
    for dep in range(len(df_copy.index)):
        for i in range(3):
            df_copy.iloc[dep, i] = X[df_copy.index[dep][1]][i]
            df_copy.iloc[dep, i+3] = Y[df_copy.index[dep][1]][i]
    return df_copy


#%%
best_value = get_value_df(best)
best_value


#%%
## worst
temp = 1
for index, row in result_read.iterrows():
    if row["TE"] < temp:
        worst = row.copy()
        temp = row["TE"]
worst


#%%
## worst data
VALUE_DATA.loc[worst.name]


#%%
class eff_collect:
    def __init__(self) -> None:
        self.TEs = []
        self.OEs = []
        self.inputs = []
        self.outputs = []
        pass
    def avg_eff(self):
        self.avg_TE = [np.average(self.TEs)]
        self.avg_OE = [np.average(self.OEs)]
        self.avg_inputs = [np.average(cell) for cell in np.array(self.inputs).T]
        self.avg_outputs = [np.average(cell) for cell in np.array(self.outputs).T]
#%%
## split by school
schools_eff = {}
for school in SCHOOLS:
    schools_eff[school] = eff_collect()
#%%
c = 0
for index, row in result_read.iterrows():
    schools_eff[index[0]].TEs.append(row["TE"])
    schools_eff[index[0]].OEs.append(row["OE"])
    schools_eff[index[0]].inputs.append(np.array(VALUE_DATA.iloc[c, 0:3]).tolist())
    schools_eff[index[0]].outputs.append(np.array(VALUE_DATA.iloc[c, 3:6]).tolist())
    c+=1
for school in SCHOOLS:
    schools_eff[school].avg_eff()
    print("avg TE of school %s is %f" %(school, schools_eff[school].avg_TE[0]))
    print("avg OE of school %s is %f" %(school, schools_eff[school].avg_OE[0]))
    print("===\n")
#%%
schools_avg = []
for key, value in schools_eff.items():
    schools_avg.append(value.avg_inputs + value.avg_outputs + value.avg_TE + value.avg_OE)
schools_avg_df = pd.DataFrame(data=schools_avg, index=schools_eff.keys(), columns=["avg_Personnel", "avg_Expenses", "avg_Space", "avg_Teaching", "avg_Publications", "avg_Grants", "avg_TE", "avg_OE"])
schools_avg_df
#%%
