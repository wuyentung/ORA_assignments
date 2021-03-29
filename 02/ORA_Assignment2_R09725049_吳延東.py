#%%
from pulp import *
import numpy as np
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

Senario = {
    "high" : 1.2, 
    "avg" : 1, 
    "low" : 0.8
    }

M = 1000000

#%%
## model
prob_a = LpProblem("max profit", LpMaximize)

## add variables
land_vars = LpVariable.dicts(name="Land", indexs=Products, lowBound=0, upBound=None, cat="continuous")
profits = LpVariable.dicts(name="profit", indexs=Products, lowBound=None, upBound=None, cat="continuous")

## objective function
prob_a += lpSum(
    [-Costs[i] * land_vars[i] for i in Products] + 
    [profits[i] for i in Products] # splitting piecewise profit function into two
    )

## constraints
prob_a += 500 >= lpSum(land_vars[i] for i in Products) # total land limit

prob_a += profits["Wheat"] <= [-np.abs(Price_below[i]) * Demands[i] + np.abs(Price_below[i]) * land_vars[i] * Avg_yield[i] for i in ["Wheat"]] # pay price when yield below demand (piecewise)
prob_a += profits["Wheat"] <= [-np.abs(Price_above[i]) * Demands[i] + np.abs(Price_above[i]) * land_vars[i] * Avg_yield[i] for i in ["Wheat"]] # get profit when yield above demand (piecewise)

prob_a += profits["Corn"] <= [-np.abs(Price_below[i]) * Demands[i] + np.abs(Price_below[i]) * land_vars[i] * Avg_yield[i] for i in ["Corn"]] # pay price when yield below demand (piecewise)
prob_a += profits["Corn"] <= [-np.abs(Price_above[i]) * Demands[i] + np.abs(Price_above[i]) * land_vars[i] * Avg_yield[i] for i in ["Corn"]] # get profit when yield above demand (piecewise)

prob_a += profits["Sugar"] <= [Price_below[i] * land_vars[i] * Avg_yield[i] for i in ["Sugar"]] # get higher profit ratio when yield below demand (piecewise)
prob_a += profits["Sugar"] <= [np.abs(Price_below[i] - Price_above[i]) * Demands[i] + Price_above[i] * land_vars[i] * Avg_yield[i] for i in ["Sugar"]] # get lower profit ratio when yield above demand (piecewise)

## solve
solve_prob(prob_a)

# https://docs.mosek.com/modeling-cookbook/mio.html
# http://civil.colorado.edu/~balajir/CVEN5393/lectures/chapter-08.pdf
#%%
'''
# RM
## expected value for each senario when yields is high, avg, or low
senario_high = LpProblem("high", LpMaximize)
high_profits = LpVariable.dicts(name="profit when high yield", indexs=Products, lowBound=0, upBound=None, cat="continuous")
## objective funciton
senario_high += lpSum([high_profits[i] for i in Products])
## constraints
senario_high += high_profits["Wheat"] >= [Price_below[i] * Demands[i] + Price_below[i] * land_vars[i] * Avg_yield[i] * Senario["high"] for i in ["Wheat"]] # pay price when yield below demand (piecewise)
senario_high += high_profits["Wheat"] <= [-Price_above[i] * Demands[i] + Price_above[i] * land_vars[i] * Avg_yield[i] * Senario["high"] for i in ["Wheat"]] # get profit when yield above demand (piecewise)

senario_high += high_profits["Corn"] >= [Price_below[i] * Demands[i] + Price_below[i] * land_vars[i] * Avg_yield[i] * Senario["high"] for i in ["Corn"]] # pay price when yield below demand (piecewise)
senario_high += high_profits["Corn"] <= [-Price_above[i] * Demands[i] + Price_above[i] * land_vars[i] * Avg_yield[i] * Senario["high"] for i in ["Corn"]] # get profit when yield above demand (piecewise)

senario_high += high_profits["Sugar"] <= [Price_below[i] * land_vars[i] * Avg_yield[i] * Senario["high"] for i in ["Sugar"]] # get higher profit ratio when yield below demand (piecewise)
senario_high += high_profits["Sugar"] <= [(Price_below[i] - Price_above[i]) * Demands[i] + Price_above[i] * land_vars[i] * Avg_yield[i] * Senario["high"] for i in ["Sugar"]] # get lower profit ratio when yield above demand (piecewise)

senario_low = LpProblem("low", LpMaximize)
low_profits = LpVariable.dicts(name="profit when high yield", indexs=Products, lowBound=0, upBound=None, cat="continuous")
## objective funciton
senario_low += lpSum([low_profits[i] for i in Products])
## constraints
senario_low += low_profits["Wheat"] >= [Price_below[i] * Demands[i] + Price_below[i] * land_vars[i] * Avg_yield[i] * Senario["low"] for i in ["Wheat"]] # pay price when yield below demand (piecewise)
senario_low += low_profits["Wheat"] <= [-Price_above[i] * Demands[i] + Price_above[i] * land_vars[i] * Avg_yield[i] * Senario["low"] for i in ["Wheat"]] # get profit when yield above demand (piecewise)

senario_low += low_profits["Corn"] >= [Price_below[i] * Demands[i] + Price_below[i] * land_vars[i] * Avg_yield[i] * Senario["low"] for i in ["Corn"]] # pay price when yield below demand (piecewise)
senario_low += low_profits["Corn"] <= [-Price_above[i] * Demands[i] + Price_above[i] * land_vars[i] * Avg_yield[i] * Senario["low"] for i in ["Corn"]] # get profit when yield above demand (piecewise)

senario_low += low_profits["Sugar"] <= [Price_below[i] * land_vars[i] * Avg_yield[i] * Senario["low"] for i in ["Sugar"]] # get higher profit ratio when yield below demand (piecewise)
senario_low += low_profits["Sugar"] <= [(Price_below[i] - Price_above[i]) * Demands[i] + Price_above[i] * land_vars[i] * Avg_yield[i] * Senario["low"] for i in ["Sugar"]] # get lower profit ratio when yield above demand (piecewise)

senario_avg = LpProblem("avg", LpMaximize)
avg_profits = LpVariable.dicts(name="profit when high yield", indexs=Products, lowBound=0, upBound=None, cat="continuous")
## objective funciton
senario_avg += lpSum([avg_profits[i] for i in Products])
## constraints
senario_avg += avg_profits["Wheat"] >= [Price_below[i] * Demands[i] + Price_below[i] * land_vars[i] * Avg_yield[i] * Senario["avg"] for i in ["Wheat"]] # pay price when yield below demand (piecewise)
senario_avg += avg_profits["Wheat"] <= [-Price_above[i] * Demands[i] + Price_above[i] * land_vars[i] * Avg_yield[i] * Senario["avg"] for i in ["Wheat"]] # get profit when yield above demand (piecewise)

senario_avg += avg_profits["Corn"] >= [Price_below[i] * Demands[i] + Price_below[i] * land_vars[i] * Avg_yield[i] * Senario["avg"] for i in ["Corn"]] # pay price when yield below demand (piecewise)
senario_avg += avg_profits["Corn"] <= [-Price_above[i] * Demands[i] + Price_above[i] * land_vars[i] * Avg_yield[i] * Senario["avg"] for i in ["Corn"]] # get profit when yield above demand (piecewise)

senario_avg += avg_profits["Sugar"] <= [Price_below[i] * land_vars[i] * Avg_yield[i] * Senario["avg"] for i in ["Sugar"]] # get higher profit ratio when yield below demand (piecewise)
senario_avg += avg_profits["Sugar"] <= [(Price_below[i] - Price_above[i]) * Demands[i] + Price_above[i] * land_vars[i] * Avg_yield[i] * Senario["avg"] for i in ["Sugar"]] # get lower profit ratio when yield above demand (piecewise)
#%%
## model
stage1 = LpProblem("stage1", LpMaximize)
## variables
stage1_land = LpVariable.dicts(name="Land", indexs=Products, lowBound=0, upBound=None, cat="continuous")
## obj
stage1 += lpSum(
    [-Costs[i] * stage1_land[i] for i in Products] + 
    [np.average([value(senario_high.objective), value(senario_avg.objective), value(senario_low.objective)])]
    )
## constraints
stage1 += lpSum([stage1_land[i] for i in Products]) <= 500
## solve


stage1.solve()
#%%
#查看目前解的狀態
print("Status:", LpStatus[stage1.status])

#印出解及目標值
for v in stage1.variables():
    print(v.name, "=", v.varValue)
print('obj=',value(stage1.objective))
'''
#%%
## model
all_stages = LpProblem("all stage merge", LpMaximize)
## vars
lands = LpVariable.dicts(name="lands for", indexs=Products, lowBound=0, upBound=None, cat="continuous")
profits = {}
for s in Senario:
    profits[s] = LpVariable.dicts(name="profit when %s yield" %s, indexs=Products, lowBound=None, upBound=None, cat="continuous")

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
