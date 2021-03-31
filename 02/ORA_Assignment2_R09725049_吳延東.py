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
## add parameters
Products = ['Wheat', 'Corn', 'Sugar']

Costs = { # per acre
    Products[0]: 150, 
    Products[1]: 230, 
    Products[2]: 260, 
    }
Avg_yield = { # average yield of crops per acre
    Products[0]: 2.5, 
    Products[1]: 3, 
    Products[2]: 20, 
    }

Demands = { # ton
    Products[0]: 200, 
    Products[1]: 240, 
    Products[2]: 6000, # set sugar price threshhold as demand
    }

Price_above = { # selling price when total yeild is above demand
    Products[0]: 170, 
    Products[1]: 150, 
    Products[2]: 10, 
    }

Price_below = { # selling price when total yeild is below demand
    Products[0]: -238, 
    Products[1]: -210, 
    Products[2]: 36, 
    }
#%%
def prob_senario_ana(prob_name="problem_s", s=1):
    ## problem
    prob_senario = LpProblem("%s" %prob_name, LpMaximize)

    ## add variables
    land_vars = LpVariable.dicts(name="Land", indexs=Products, lowBound=0, upBound=None, cat="continuous")
    profits = LpVariable.dicts(name="profit", indexs=Products, lowBound=None, upBound=None, cat="continuous")

    ## objective function
    prob_senario += lpSum(
        [-Costs[i] * land_vars[i] for i in Products] + 
        [profits[i] for i in Products] # splitting piecewise profit function into two
        )

    ## constraints
    prob_senario += 500 >= lpSum(land_vars[i] for i in Products) # total land limit

    prob_senario += profits["Wheat"] <= [-np.abs(Price_below[i]) * Demands[i] + np.abs(Price_below[i]) * land_vars[i] * Avg_yield[i] * s for i in ["Wheat"]] # pay price when yield below demand (piecewise)
    prob_senario += profits["Wheat"] <= [-np.abs(Price_above[i]) * Demands[i] + np.abs(Price_above[i]) * land_vars[i] * Avg_yield[i] * s for i in ["Wheat"]] # get profit when yield above demand (piecewise)

    prob_senario += profits["Corn"] <= [-np.abs(Price_below[i]) * Demands[i] + np.abs(Price_below[i]) * land_vars[i] * Avg_yield[i] * s for i in ["Corn"]] # pay price when yield below demand (piecewise)
    prob_senario += profits["Corn"] <= [-np.abs(Price_above[i]) * Demands[i] + np.abs(Price_above[i]) * land_vars[i] * Avg_yield[i] * s for i in ["Corn"]] # get profit when yield above demand (piecewise)

    prob_senario += profits["Sugar"] <= [Price_below[i] * land_vars[i] * Avg_yield[i] * s for i in ["Sugar"]] # get higher profit ratio when yield below demand (piecewise)
    prob_senario += profits["Sugar"] <= [np.abs(Price_below[i] - Price_above[i]) * Demands[i] + Price_above[i] * land_vars[i] * Avg_yield[i] * s for i in ["Sugar"]] # get lower profit ratio when yield above demand (piecewise)

    ## solve
    prob_senario.solve()
    return prob_senario
#%%
def prob_result(prob):
    print("Status for %s:" %prob.name, LpStatus[prob.status])

    #印出解及目標值
    for v in prob.variables():
        print(v.name, "=", v.varValue)
    print('obj=',value(prob.objective))
    pass
#%%
def senario_obj(determined_land, s=1):
    cost = np.sum([determined_land[i] * Costs[i] for i in Products])

    profits = {
        Products[0]: 0,
        Products[1]: 0,
        Products[2]: 0,
    }

    for i in Products:
        y = determined_land[i] * Avg_yield[i] * s
        if y <= Demands[i]:
            if i != "Sugar":
                profits[i] = Price_below[i] * (Demands[i] - y)
            else:
                profits[i] = Price_below[i] * y
        else:
            if i != "Sugar":
                profits[i] = Price_above[i] * (y - Demands[i])
            else:
                profits[i] = Price_below[i] * y
    return np.sum([profits[i] for i in Products]) - cost
#%%
## 1a
prob_a = prob_senario_ana(prob_name="1a")
prob_result(prob=prob_a)

# https://docs.mosek.com/modeling-cookbook/mio.html
# http://civil.colorado.edu/~balajir/CVEN5393/lectures/chapter-08.pdf
#%%
## 1b
Senario = {
    "high" : 1.2, 
    "avg" : 1, 
    "low" : 0.8
    }
#%%
prob_1bs = {}
for s in Senario:
    prob_1bs[s] = prob_senario_ana(prob_name="1b_%s" %s, s=Senario[s])
    prob_result(prob=prob_1bs[s])
    print()

#%%
## 1d
## model
all_stages = LpProblem("problem_1c", LpMaximize)
## vars
lands = LpVariable.dicts(name="lands_for", indexs=Products, lowBound=0, upBound=None, cat="continuous")
profits = {}
for s in Senario:
    profits[s] = LpVariable.dicts(name="profit_when_%s_yield" %s, indexs=Products, lowBound=None, upBound=None, cat="continuous")

## obj
all_stages += lpSum([-Costs[i] * lands[i] for i in Products]) + 1/3 * lpSum([profits[s][i] for s in Senario for i in Products])

## st
### land
all_stages += 500 >= lpSum([lands[i] for i in Products])

### 
for s in Senario:
    all_stages += profits[s]["Wheat"] <= [-np.abs(Price_below[i]) * Demands[i] + np.abs(Price_below[i]) * lands[i] * Avg_yield[i] * Senario[s] for i in ["Wheat"]] # pay price when yield below demand (piecewise)
    all_stages += profits[s]["Wheat"] <= [-np.abs(Price_above[i]) * Demands[i] + np.abs(Price_above[i]) * lands[i] * Avg_yield[i] * Senario[s] for i in ["Wheat"]] # get profit when yield above demand (piecewise)

    all_stages += profits[s]["Corn"] <= [-np.abs(Price_below[i]) * Demands[i] + np.abs(Price_below[i]) * lands[i] * Avg_yield[i] * Senario[s] for i in ["Corn"]] # pay price when yield below demand (piecewise)
    all_stages += profits[s]["Corn"] <= [-np.abs(Price_above[i]) * Demands[i] + np.abs(Price_above[i]) * lands[i] * Avg_yield[i] * Senario[s] for i in ["Corn"]] # get profit when yield above demand (piecewise)

    all_stages += profits[s]["Sugar"] <= [Price_below[i] * lands[i] * Avg_yield[i] * Senario[s] for i in ["Sugar"]] # get higher profit ratio when yield below demand (piecewise)
    all_stages += profits[s]["Sugar"] <= [np.abs(Price_below[i] - Price_above[i]) * Demands[i] + Price_above[i] * lands[i] * Avg_yield[i] * Senario[s] for i in ["Sugar"]] # get lower profit ratio when yield above demand (piecewise)

solve_prob(all_stages)
#%%
land_1d = {}
land_1d[Products[0]] = all_stages.variables()[2].varValue
land_1d[Products[1]] = all_stages.variables()[0].varValue
land_1d[Products[2]] = all_stages.variables()[1].varValue
for s in Senario:
    print("1d for %s demand: " % s , senario_obj(determined_land=land_1d, s=Senario[s]))
    print()
#%%
## 1e EVPI, VSS
### EVPI
## max: WS - RP
ws = np.mean([value(prob_1bs[s].objective) for s in Senario])
rp = value(all_stages.objective)
evpi = ws - rp
### VSS = EEV - RP

#%%
## 1g
## set samples
mu, sigma = 1, 0.1
M = []
T = []
for i in range(15):
    M.append(np.random.normal(mu, sigma, 30))
    T.append(np.random.normal(mu, sigma, 30))


#%%
low_vars = []
low_obj = []
for i in range(15):
    low_vars.append([None] * 30)
    low_obj.append([None] * 30)

for i in range(15):
    for j in range(30):
        low_prob = prob_senario_ana(M[i][j])
        low_vars[i][j], low_obj[i][j] = low_prob.variables(), value(low_prob.objective)
#%%
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
#%%
lower_bound = {}
lower_bound["mean"], lower_bound["low"], lower_bound["high"] = mean_confidence_interval(data=[low_obj[i][j] for i in range(15) for j in range(30)])
#%%
## upper
land_mu = {}
land_mu["Wheat"] = np.mean([low_vars[i][j][2].varValue for i in range(15) for j in range(30)])
land_mu["Corn"] = np.mean([low_vars[i][j][0].varValue for i in range(15) for j in range(30)])
land_mu["Sugar"] = np.mean([low_vars[i][j][1].varValue for i in range(15) for j in range(30)])
#%%
up_obj = []
for i in range(15):
    up_obj.append([None] * 30)

for i in range(15):
    for j in range(30):
        up_obj[i][j] = senario_obj(determined_land=land_mu, s=T[i][j])
#%%
upper_bound = {}
upper_bound["mean"], upper_bound["low"], upper_bound["high"] = mean_confidence_interval(data=[up_obj[i][j] for i in range(15) for j in range(30)])

#%%
print("lower_bound: ", lower_bound)
print("upper_bound: ", upper_bound)
#%%
