#%%
from pulp import *
#%%
## add parameters
Products = ['Wheat', 'Corn', 'Sugar']

costs = { # per acre
    Products[0]: 150, 
    Products[1]: 230, 
    Products[2]: 260, 
    }
avg_yield = { # average yield of crops per acre
    Products[0]: 2.5, 
    Products[1]: 3, 
    Products[2]: 20, 
    }

demands = { # ton
    Products[0]: 200, 
    Products[1]: 240, 
    Products[2]: 6000, # set sugar price threshhold as demand
    }

price_above = { # selling price when total yeild is above demand
    Products[0]: 170, 
    Products[1]: 150, 
    Products[2]: 10, 
    }

price_below = { # selling price when total yeild is below demand
    Products[0]: -238, 
    Products[1]: -210, 
    Products[2]: 36, 
    }

#%%
## model
prob_a = LpProblem("lunch cost down", LpMaximize)
#%%
## decision vars
land_vars = LpVariable.dicts(name="Land", indexs=Products, lowBound=0, upBound=None, cat="continuous")
products_above = LpVariable.dicts(name="products_above", indexs=Products, lowBound=0, upBound=None, cat="continuous")
products_below = LpVariable.dicts(name="products_below", indexs=Products, lowBound=0, upBound=None, cat="continuous")
#%%
## objective function
prob_a += lpSum(
    [price_above[i] * products_above[i] for i in Products]
    + [price_below[i] * products_below[i] for i in Products]
    + [-costs[i] * land_vars[i] for i in Products]
    )
#%%
## constraints
prob_a += 500 >= lpSum([land_vars[i] for i in Products])
prob_a += 0 == [avg_yield[i] * land_vars[i] - products_above[i] + products_below[i] - demands[i] for i in ["Wheat"]][0]
prob_a += 0 == [avg_yield[i] * land_vars[i] - products_above[i] + products_below[i] - demands[i] for i in ["Corn"]][0]
prob_a += 0 == [avg_yield[i] * land_vars[i] - products_above[i] - products_below[i] for i in ["Sugar"]][0]
prob_a += 0 <= [demands[i] - products_below[i] for i in ["Sugar"]][0]

#%%
## solve
prob_a.solve()
#%%
#查看目前解的狀態
print("Status:", LpStatus[prob_a.status])

#印出解及目標值
for v in prob_a.variables():
    print(v.name, "=", v.varValue)
print('obj=',value(prob_a.objective))
#解的另一種方式
# for i in Ingredients:
#   print(ingredient_vars[i],"=",ingredient_vars[i].value())
#%%
import numpy as np
#%%
p = np.array([[0.8, 0.2, 0, 0], [0, 0, 0.2, 0.8], [0, 1, 0, 0], [0.8, 0.2, 0, 0]])
print(p)
# %%
p2 = np.dot(p, p)
print(p2)
print(p2[0][2])
# %%
p4 = np.dot(p2, p2)
p5 = np.dot(p4, p)
print(p5)
print(p5[0][2])
# %%
p10 = np.dot(p5, p5)
print(p10)
print(p10[0][2])
# %%
p20 = np.dot(p10, p10)
print(p20)
print(p20[0][2])
# %%
p40 = np.dot(p20, p20)
print(p40)
print(p40[0][2])
# %%
inoperable_cost = 30000
p_inoperable = p40[0][2] 
expected_cost = p_inoperable * inoperable_cost + (1 - p_inoperable) * 0
print(expected_cost)
# %%
