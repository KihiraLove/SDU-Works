import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum


def time_to_minutes(time :str) -> int:
    h, m = map(int, str(time).split(':'))
    return h * 60 + m


# Load data
df_coordinates = pd.read_csv('data/instance_coordinates.txt', sep=r'\s+', header=None, names=['id', 'x', 'y'])
df_demands = pd.read_csv('data/instance_demand.txt', sep=r'\s+', header=None, names=['id', 'demand'])
df_service_times = pd.read_csv('data/instance_service_time.txt', sep=r'\s+', header=None, names=['id', 'service'])
df_time_windows_raw = pd.read_csv('data/instance_time_windows.txt', sep=r'\s+', header=None, names=['id', 'start', 'end'])

df_time_windows_raw['start'] = df_time_windows_raw['start'].apply(time_to_minutes)
df_time_windows_raw['end'] = df_time_windows_raw['end'].apply(time_to_minutes)

# Merge data
data = df_coordinates.merge(df_demands, on='id').merge(df_service_times, on='id')
data = data.sort_values(by='id').reset_index(drop=True)
num_nodes = len(data)

# Distance and travel time matrix
locations = data[['x', 'y']].to_numpy()
dist_matrix = np.linalg.norm(locations[:, None] - locations[None, :], axis=2)
vehicle_speed = 1000 / 60  # units per minute
travel_time_matrix = dist_matrix / vehicle_speed

# Time windows: merge all intervals into one (earliest start to latest end)
raw_time_windows = {i: [] for i in data['id']}
for _, row in df_time_windows_raw.iterrows():
    raw_time_windows[int(row['id'])].append((int(row['start']), int(row['end'])))

consolidated_time_windows = [
    (min(tws, key=lambda x: x[0])[0], max(tws, key=lambda x: x[1])[1])
    for i, tws in raw_time_windows.items()
]

# Check for infeasible demands
assert all(data['demand'] <= 12600), "Some customer demands exceed vehicle capacity"

# Model
model = Model('VRPTW')
vehicle_capacity = 12600
bigM = 1440
penalty = 1e6

arc_used = model.addVars(num_nodes, num_nodes, vtype=GRB.BINARY, name='x')
arrival_time = model.addVars(num_nodes, vtype=GRB.CONTINUOUS, name='t')
load_after_service = model.addVars(num_nodes, vtype=GRB.CONTINUOUS, name='load')
mtz_order = model.addVars(num_nodes, vtype=GRB.CONTINUOUS, name='u')  # for subtour elimination (MTZ)
served = model.addVars(range(1, num_nodes), vtype=GRB.BINARY, name='z')  # serve indicator

# Objective: minimize total distance + penalty for skipping customers
model.setObjective(
    quicksum(dist_matrix[i][j] * arc_used[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j) +
    penalty * quicksum(1 - served[i] for i in range(1, num_nodes)), GRB.MINIMIZE
)

# Routing constraints with serve indicator
for i in range(1, num_nodes):
    model.addConstr(quicksum(arc_used[j, i] for j in range(num_nodes) if j != i) == served[i])
    model.addConstr(quicksum(arc_used[i, j] for j in range(num_nodes) if j != i) == served[i])

model.addConstr(quicksum(arc_used[0, j] for j in range(1, num_nodes)) >= 1)
model.addConstr(quicksum(arc_used[i, 0] for i in range(1, num_nodes)) >= 1)

# New: Ensure depot connections match inbound/outbound
model.addConstr(quicksum(arc_used[0, j] for j in range(1, num_nodes)) <= num_nodes)
model.addConstr(quicksum(arc_used[i, 0] for i in range(1, num_nodes)) <= num_nodes)
for j in range(1, num_nodes):
    model.addConstr(arc_used[0, j] <= quicksum(arc_used[j, k] for k in range(num_nodes) if k != j))
    model.addConstr(arc_used[j, 0] <= quicksum(arc_used[i, j] for i in range(num_nodes) if i != j))

# Prevent depot self-loop
arc_used[0, 0].ub = 0

# Time window constraints
for i in range(num_nodes):
    start, end = consolidated_time_windows[i]
    model.addConstr(arrival_time[i] >= start)
    model.addConstr(arrival_time[i] <= end)

# Depot time and load initialization
model.addConstr(arrival_time[0] >= 0)
model.addConstr(arrival_time[0] <= 1440)
model.addConstr(load_after_service[0] == 0)

# Capacity constraints (skip depot)
for i in range(1, num_nodes):
    model.addConstr(load_after_service[i] >= data.loc[i, 'demand'] * served[i])
    model.addConstr(load_after_service[i] <= vehicle_capacity)

# ====== ONLY TIME PROPAGATION, SAFELY GUARDED ======
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            # Only enforce if both i and j are to be served
            if i > 0 and j > 0:
                model.addConstr(
                    arrival_time[j] >= arrival_time[i] + travel_time_matrix[i][j] + data.loc[i, 'service']
                    - bigM * (1 - arc_used[i, j] + (1 - served[i]) + (1 - served[j]))
                )

# ====== LOAD PROPAGATION, SAFELY GUARDED ======
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            if i > 0 and j > 0:
                model.addConstr(
                    load_after_service[j] >= load_after_service[i] + data.loc[j, 'demand']
                    - vehicle_capacity * (1 - arc_used[i, j] + (1 - served[i]) + (1 - served[j]))
                )

# Subtour elimination (MTZ) - skip depot
for i in range(1, num_nodes):
    model.addConstr(mtz_order[i] >= 1)
    model.addConstr(mtz_order[i] <= num_nodes - 1)
for i in range(1, num_nodes):
    for j in range(1, num_nodes):
        model.addConstr(mtz_order[i] - mtz_order[j] + (num_nodes - 1) * arc_used[i, j] <= num_nodes - 2 + (1 - served[i]) * num_nodes + (1 - served[j]) * num_nodes)

model.setParam('TimeLimit', 300)
model.optimize()

# Output solution or diagnose infeasibility
if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
    print(f"Total distance (with penalties): {model.objVal:.2f}")
    solution = model.getAttr('x', arc_used)
    for i in range(num_nodes):
        for j in range(num_nodes):
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

for start in range(num_nodes):
    if start == 0:
        for j in range(num_nodes):
            try:
                if arc_used[start, j].X > 0.5 and j != start:
                    route = [start]
                    times = [arrival_time[start].X]
                    loads = [load_after_service[start].X]
                    curr = j
                    distance = dist_matrix[start][j]
                    while curr != 0 and curr not in route:
                        route.append(curr)
                        times.append(arrival_time[curr].X)
                        loads.append(load_after_service[curr].X)
                        next_found = False
                        for k in range(num_nodes):
                            if arc_used[curr, k].X > 0.5 and k != curr:
                                distance += dist_matrix[curr][k]
                                curr = k
                                next_found = True
                                break
                        if not next_found:
                            break
                    route.append(0)
                    times.append(arrival_time[0].X)
                    loads.append(load_after_service[0].X)
                    if len(route) > 2 and tuple(route) not in visited_routes:
                        routes.append((route, times, loads, df_service_times))
                        route_distances.append(distance)
                        visited_routes.add(tuple(route))
            except Exception:
                pass

# Print all routes
print(f"Number of vehicles used: {len(routes)}\n")
total_distance = 0
for idx, (route, times, loads, df_service_times) in enumerate(routes):
    print(f"Vehicle {idx+1} route (Distance: {round(route_distances[idx], 2)}):")
    total_distance += route_distances[idx]
    for node, time, load_after_service, stime in zip(route, times, loads, df_service_times['service']):
        print(f"  Node {node:3d}: Time={round(time,2):6.2f}, Load={round(load_after_service, 2):6.2f}, Service={round(int(stime), 2):6.2f}")
    print()
print(f"Total distance: {round(total_distance,2)}")
