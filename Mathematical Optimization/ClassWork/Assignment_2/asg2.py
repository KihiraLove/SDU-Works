import pandas as pd
import numpy as np
from math import sqrt
from gurobipy import Model, GRB, quicksum

# Load data
coordinates = pd.read_csv('instance_coordinates.txt', sep=r'\s+', header=None, names=['id', 'x', 'y'])
demands = pd.read_csv('instance_demand.txt', sep=r'\s+', header=None, names=['id', 'demand'])
service_times = pd.read_csv('instance_service_time.txt', sep=r'\s+', header=None, names=['id', 'service'])
time_windows_raw = pd.read_csv('instance_time_windows.txt', sep=r'\s+', header=None, names=['id', 'start', 'end'])
# Time conversion
def time_to_minutes(t):
    h, m = map(int, str(t).split(':'))
    return h * 60 + m

time_windows_raw['start'] = time_windows_raw['start'].apply(time_to_minutes)
time_windows_raw['end'] = time_windows_raw['end'].apply(time_to_minutes)

# Merge data
data = coordinates.merge(demands, on='id').merge(service_times, on='id')
data = data.sort_values(by='id').reset_index(drop=True)
n = len(data)

# Distance and travel time matrix
coords = data[['x', 'y']].to_numpy()
dist = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
speed = 1000 / 60  # units per minute
travel_time = dist / speed

# Time windows: merge all intervals into one (earliest start to latest end)
time_windows = {i: [] for i in data['id']}
for _, row in time_windows_raw.iterrows():
    time_windows[int(row['id'])].append((int(row['start']), int(row['end'])))

simplified_windows = [
    (min(tws, key=lambda x: x[0])[0], max(tws, key=lambda x: x[1])[1])
    for i, tws in time_windows.items()
]

# Check for infeasible demands
assert all(data['demand'] <= 12600), "Some customer demands exceed vehicle capacity"

# Model
model = Model('VRPTW')
Q = 12600  # vehicle capacity
bigM = 1440
penalty = 1e6

x = model.addVars(n, n, vtype=GRB.BINARY, name='x')
t = model.addVars(n, vtype=GRB.CONTINUOUS, name='t')
load = model.addVars(n, vtype=GRB.CONTINUOUS, name='load')
u = model.addVars(n, vtype=GRB.CONTINUOUS, name='u')  # for subtour elimination (MTZ)
z = model.addVars(range(1, n), vtype=GRB.BINARY, name='z')  # serve indicator

# Objective: minimize total distance + penalty for skipping customers
model.setObjective(
    quicksum(dist[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j) +
    penalty * quicksum(1 - z[i] for i in range(1, n)), GRB.MINIMIZE
)

# Routing constraints with serve indicator
for i in range(1, n):
    model.addConstr(quicksum(x[j, i] for j in range(n) if j != i) == z[i])
    model.addConstr(quicksum(x[i, j] for j in range(n) if j != i) == z[i])

model.addConstr(quicksum(x[0, j] for j in range(1, n)) >= 1)
model.addConstr(quicksum(x[i, 0] for i in range(1, n)) >= 1)

# New: Ensure depot connections match inbound/outbound
model.addConstr(quicksum(x[0, j] for j in range(1, n)) <= n)
model.addConstr(quicksum(x[i, 0] for i in range(1, n)) <= n)
for j in range(1, n):
    model.addConstr(x[0, j] <= quicksum(x[j, k] for k in range(n) if k != j))
    model.addConstr(x[j, 0] <= quicksum(x[i, j] for i in range(n) if i != j))

# Prevent depot self-loop
x[0, 0].ub = 0

# Time window constraints
for i in range(n):
    start, end = simplified_windows[i]
    model.addConstr(t[i] >= start)
    model.addConstr(t[i] <= end)

# Depot time and load initialization
model.addConstr(t[0] >= 0)
model.addConstr(t[0] <= 1440)
model.addConstr(load[0] == 0)

# Capacity constraints (skip depot)
for i in range(1, n):
    model.addConstr(load[i] >= data.loc[i, 'demand'] * z[i])
    model.addConstr(load[i] <= Q)

# ====== ONLY TIME PROPAGATION, SAFELY GUARDED ======
for i in range(n):
    for j in range(n):
        if i != j:
            # Only enforce if both i and j are to be served
            if i > 0 and j > 0:
                model.addConstr(
                    t[j] >= t[i] + travel_time[i][j] + data.loc[i, 'service']
                    - bigM * (1 - x[i, j] + (1 - z[i]) + (1 - z[j]))
                )

# ====== LOAD PROPAGATION, SAFELY GUARDED ======
for i in range(n):
    for j in range(n):
        if i != j:
            if i > 0 and j > 0:
                model.addConstr(
                    load[j] >= load[i] + data.loc[j, 'demand']
                    - Q * (1 - x[i, j] + (1 - z[i]) + (1 - z[j]))
                )

# Subtour elimination (MTZ) - skip depot
for i in range(1, n):
    model.addConstr(u[i] >= 1)
    model.addConstr(u[i] <= n - 1)
for i in range(1, n):
    for j in range(1, n):
        model.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2 + (1 - z[i]) * n + (1 - z[j]) * n)

model.setParam('TimeLimit', 300)
model.optimize()

# Output solution or diagnose infeasibility
if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
    print(f"Total distance (with penalties): {model.objVal:.2f}")
    solution = model.getAttr('x', x)
    for i in range(n):
        for j in range(n):
            if i != j and solution[i, j] > 0.5:
                print(f"Route: {i} -> {j}")
elif model.status == GRB.INFEASIBLE:
    print("Model infeasible. Computing IIS...")
    model.computeIIS()
    model.write("model.ilp")
    print("IIS written to model.ilp")
else:
    print("Model could not be solved or no solution found.")

# Assumptions:
# - x[i, j]: binary, 1 if vehicle uses arc i->j
# - t[i]: time of arrival/start at node i
# - l[i]: load of vehicle after servicing node i
# - s[i]: service time at node i
# - dist[i][j]: distance from node i to node j
# - n: number of nodes (including depot 0)

routes = []
route_distances = []
visited_routes = set()

for start in range(n):
    if start == 0:
        for j in range(n):
            try:
                if x[start, j].X > 0.5 and j != start:
                    route = [start]
                    times = [t[start].X]
                    loads = [load[start].X]
                    curr = j
                    distance = dist[start][j]
                    while curr != 0 and curr not in route:
                        route.append(curr)
                        times.append(t[curr].X)
                        loads.append(load[curr].X)
                        next_found = False
                        for k in range(n):
                            if x[curr, k].X > 0.5 and k != curr:
                                distance += dist[curr][k]
                                curr = k
                                next_found = True
                                break
                        if not next_found:
                            break
                    route.append(0)
                    times.append(t[0].X)
                    loads.append(load[0].X)
                    if len(route) > 2 and tuple(route) not in visited_routes:
                        routes.append((route, times, loads, service_times))
                        route_distances.append(distance)
                        visited_routes.add(tuple(route))
            except Exception:
                pass

# Print all routes
print(f"Number of vehicles used: {len(routes)}\n")
total_distance = 0
for idx, (route, times, loads, service_times) in enumerate(routes):
    print(f"Vehicle {idx+1} route (Distance: {round(route_distances[idx], 2)}):")
    total_distance += route_distances[idx]
    for node, time, load, stime in zip(route, times, loads, service_times['service']):
        print(f"  Node {node:3d}: Time={round(time,2):6.2f}, Load={round(load,2):6.2f}, Service={round(int(stime),2):6.2f}")
    print()
print(f"Total distance: {round(total_distance,2)}")
