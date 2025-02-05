from gurobipy import *

# %%
# Gurobi modelling 1
m = Model()

x = m.addVar(vtype=GRB.BINARY, name="x")
y = m.addVar(vtype=GRB.BINARY, name="y")
z = m.addVar(vtype=GRB.BINARY, name="z")

m.setObjective(x + y + 2*z, GRB.MAXIMIZE)

c1 = m.addConstr(x + 2*y + 4*z <= 4)
c2 = m.addConstr(x + y >= 1)

m.optimize()

# %%
# Gurobi python functions

# list comprehension
sqrd = [i*i for i in range(5)]
print(sqrd)

# subsequence with condition
bigsqrd = [i*i for i in range(5) if i*i >= 5]
print(bigsqrd) # displays [9, 16]

# multiple for loops
prod = [i*j for i in range(3) for j in range(4)]
print(prod) # displays [0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6]

# generator expression
sumsqrd = sum(i*i for i in range(5))
print(sumsqrd) # displays 30

# %%
# Gurobi modelling 2

# %%
# explicit deleting of model
del m