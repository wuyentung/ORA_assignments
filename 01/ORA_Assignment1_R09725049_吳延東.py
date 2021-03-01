#%%
from pulp import *
#%%
## add parameters
Ingredients = ['BREAD', 'PEANUT BUTTER', 'JELLY', 'CRACKER', 'MILK', 'JUICE']

costs = { # cents
    Ingredients[0]: 5, 
    Ingredients[1]: 4, 
    Ingredients[2]: 7, 
    Ingredients[3]: 8, 
    Ingredients[4]: 15, 
    Ingredients[5]: 35}

fat_kcals = { # kcal
    Ingredients[0]: 10, 
    Ingredients[1]: 75, 
    Ingredients[2]: 0, 
    Ingredients[3]: 20, 
    Ingredients[4]: 70, 
    Ingredients[5]: 0}

total_kcals = { # kcal
    Ingredients[0]: 70, 
    Ingredients[1]: 100, 
    Ingredients[2]: 50, 
    Ingredients[3]: 60, 
    Ingredients[4]: 150, 
    Ingredients[5]: 100}

vit_C = { # mg
    Ingredients[0]: 0, 
    Ingredients[1]: 0, 
    Ingredients[2]: 3, 
    Ingredients[3]: 0, 
    Ingredients[4]: 2, 
    Ingredients[5]: 120}

protein = { # g
    Ingredients[0]: 3, 
    Ingredients[1]: 4, 
    Ingredients[2]: 0, 
    Ingredients[3]: 1, 
    Ingredients[4]: 8, 
    Ingredients[5]: 1}

#%%
## model
prob = LpProblem("lunch cost down", LpMinimize)
#%%
## decision vars
ingredient_vars = LpVariable.dicts(name="Ingredient", indexs=Ingredients, lowBound=0, cat="Integer")
#%%
## objective function
prob += lpSum([costs[i] * ingredient_vars[i] for i in Ingredients])
#%%
## constraints, nutritional requirements
prob += lpSum([total_kcals[i] for i in Ingredients]) >= 400 # total kcal
prob += lpSum([total_kcals[i] for i in Ingredients]) <= 600 # total kcal
prob += (lpSum([fat_kcals[i] for i in Ingredients]) / lpSum([total_kcals[i] for i in Ingredients])) <= 0.3 # proportion of fat kcal
prob += lpSum([vit_C[i] * ingredient_vars[i] for i in Ingredients]) >= 60 # at least 60 mg vitamin C
prob += lpSum([protein[i] * ingredient_vars[i] for i in Ingredients]) >= 12 # at least 12 g protein
prob += ingredient_vars["BREAD"] == 2 # 2 slices of bread
prob += ingredient_vars["PEANUT BUTTER"] >= 2 * ingredient_vars["JELLY"] # peanut butter is two times of jelly
prob += ingredient_vars["MILK"] + ingredient_vars["JUICE"] >= 1 # at least one cup of liquid
#%%
## solve
prob.solve()
#%%
#查看目前解的狀態
print("Status:", LpStatus[prob.status])

#印出解及目標值
for v in prob.variables():
    print(v.name, "=", v.varValue)
print('obj=',value(prob.objective))
#解的另一種方式
# for i in Ingredients:
#   print(ingredient_vars[i],"=",ingredient_vars[i].value())
# %%
