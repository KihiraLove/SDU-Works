import pandas as pd
import numpy as np

from gurobipy import Model, GRB, quicksum

# Load data
coordinates = pd.read_csv('data/instance_coordinates.txt', sep=' ', header=None, names=['id', 'x', 'y'])
demands = pd.read_csv('data/instance_demand.txt', sep=' ', header=None, names=['id', 'demand'])
service_times = pd.read_csv('data/instance_service_time.txt', sep=' ', header=None, names=['id', 'service'])
time_windows_raw = pd.read_csv('data/instance_time_windows.txt', sep=' ', header=None, names=['id', 'start', 'end'])

# Time conversion
def time_to_minutes(t):
    h, m = map(int, t.split(':'))
    return h * 60 + m

time_windows_raw['start'] = time_windows_raw['start'].apply(lambda x: time_to_minutes(str(x)))
time_windows_raw['end'] = time_windows_raw['end'].apply(lambda x: time_to_minutes(str(x)))

# Merge data
data = coordinates.merge(demands, on='id').merge(service_times, on='id')
data = data.sort_values(by='id').reset_index(drop=True)
n = len(data)

# Distance matrix
coords = data[['x', 'y']].to_numpy()
dist = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
speed = 1000 / 60  # distance units per minute
travel_time = dist / speed

# Time windows
time_windows = {i: [] for i in data['id']}
for _, row in time_windows_raw.iterrows():
    time_windows[int(row['id'])].append((int(row['start']), int(row['end'])))

# Simplify time windows by using the earliest interval only
simplified_windows = [min(time_windows.get(i+1, [(0, 1440)])) for i in range(n)]

# Model
model = Model('VRPTW')
Q = 12600  # vehicle capacity
bigM = 1e6

x = model.addVars(n, n, vtype=GRB.BINARY, name='x')
t = model.addVars(n, vtype=GRB.CONTINUOUS, name='t')
load = model.addVars(n, vtype=GRB.CONTINUOUS, name='load')
u = model.addVars(n, vtype=GRB.CONTINUOUS, name='u')  # for subtour elimination (MTZ)

# Objective: minimize total distance
model.setObjective(quicksum(dist[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j), GRB.MINIMIZE)

# Constraints
for i in range(1, n):
    model.addConstr(quicksum(x[j, i] for j in range(n) if j != i) == 1)
    model.addConstr(quicksum(x[i, j] for j in range(n) if j != i) == 1)

model.addConstr(quicksum(x[0, j] for j in range(1, n)) >= 1)
model.addConstr(quicksum(x[i, 0] for i in range(1, n)) >= 1)

for i in range(n):
    start, end = simplified_windows[i]
    model.addConstr(t[i] >= start)
    model.addConstr(t[i] <= end)

for i in range(n):
    model.addConstr(load[i] >= data.loc[i, 'demand'])
    model.addConstr(load[i] <= Q)

for i in range(n):
    for j in range(n):
        if i != j:
            model.addConstr(t[j] >= t[i] + travel_time[i][j] + data.loc[i, 'service'] - bigM * (1 - x[i, j]))
            model.addConstr(load[j] >= load[i] + data.loc[j, 'demand'] - Q * (1 - x[i, j]))

# Subtour elimination (MTZ)
for i in range(1, n):
    model.addConstr(u[i] >= 1)
    model.addConstr(u[i] <= n - 1)
for i in range(1, n):
    for j in range(1, n):
        if i != j:
            model.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2)

model.setParam('TimeLimit', 300)
model.optimize()

# Print results
if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
    print(f"Total distance: {model.objVal:.2f}")
    solution = model.getAttr('x', x)
    for i in range(n):
        for j in range(n):
            if i != j and solution[i, j] > 0.5:
                print(f"Route: {i} -> {j}")
