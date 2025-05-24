import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum


def time_to_minutes(time_str: str) -> int:
    r"""
    Convert a time string to minutes since midnight.

    :param time_str: Time formatted as "HH:MM".
    :return: Integer minutes since midnight.
    """
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes


def load_data() -> tuple:
    r"""
    Load data from text files.
    :return: Tuple of DataFrames (coordinates, demands, service_times, time_windows_raw).
    """
    coord_path = 'data/instance_coordinates.txt'
    demand_path = 'data/instance_demand.txt'
    service_path = 'data/instance_service_time.txt'
    tw_path = 'data/instance_time_windows.txt'

    # Load coordinate and attribute files into DataFrames
    df_coordinates = pd.read_csv(coord_path, sep=r'\s+', header=None, names=['id', 'x', 'y'])
    df_demands = pd.read_csv(demand_path, sep=r'\s+', header=None, names=['id', 'demand'])
    df_service_times = pd.read_csv(service_path, sep=r'\s+', header=None, names=['id', 'service'])
    df_time_windows_raw = pd.read_csv(tw_path, sep=r'\s+', header=None, names=['id', 'start', 'end'])

    # Normalize time window endpoints to integer minutes
    df_time_windows_raw['start'] = df_time_windows_raw['start'].apply(time_to_minutes)
    df_time_windows_raw['end'] = df_time_windows_raw['end'].apply(time_to_minutes)

    return df_coordinates, df_demands, df_service_times, df_time_windows_raw


def merge_data(df_coordinates: pd.DataFrame, df_demands: pd.DataFrame, df_service_times: pd.DataFrame) -> pd.DataFrame:
    r"""
    Merge coordinate, demand, and service time DataFrames into a single table.

    :param df_coordinates: DataFrame of node coordinates.
    :param df_demands: DataFrame of demands.
    :param df_service_times: DataFrame of service times.
    :return: Merged DataFrame sorted by node id.
    """
    df = (
        df_coordinates
        .merge(df_demands, on='id')
        .merge(df_service_times, on='id')
        .sort_values('id') # sort to ascending id order (0 is depot)
        .reset_index(drop=True)
    )
    return df


def compute_matrices(df_instances: pd.DataFrame) -> tuple:
    r"""
    Compute pairwise distance and travel-time matrices.
    - distance_matrix[i][j]: Euclidean distance between nodes i and j.
    - travel_time_matrix: distance divided by constant vehicle speed.
    :param df_instances: DataFrame with x,y columns.
    :return: distance_matrix, travel_time_matrix.
    """
    coords = df_instances[['x', 'y']].to_numpy()
    # Euclidean distance with numpy
    distance_matrix = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)

    # Convert to travel times assuming uniform speed
    speed = 1000 / 60
    travel_time_matrix = distance_matrix / speed
    return distance_matrix, travel_time_matrix


def consolidate_time_windows(df_time_windows_raw: pd.DataFrame, ids: np.ndarray) -> list:
    r"""
    Merge possibly multiple time window intervals into a single [earliest, latest] per node.
    Some instances list disjoint intervals. For simplicity, we take the global min start and max end.

    :param df_time_windows_raw: DataFrame with raw intervals.
    :param ids: Array of node IDs.
    :return: List of (earliest_start, latest_end) tuples.
    """
    raw = {int(i): [] for i in ids}
    # Group all intervals by node id
    for _, row in df_time_windows_raw.iterrows():
        raw[int(row['id'])].append((row['start'], row['end']))
    consolidated = []
    for i in ids:
        intervals = raw[int(i)]
        # Determine the earliest possible start and latest end
        start = min(intervals, key=lambda x: x[0])[0]
        end = max(intervals, key=lambda x: x[1])[1]
        consolidated.append((start, end))
    return consolidated


def check_capacity(df_instances: pd.DataFrame, capacity: float) -> None:
    r"""
    Ensure all demands are within vehicle capacity.

    :param df_instances: DataFrame with demand column.
    :param capacity: Vehicle capacity.
    :raises AssertionError: if any demand exceeds capacity.
    """
    assert (df_instances['demand'] <= capacity).all(), "Customer demand exceeds capacity"


def build_model(num_nodes: int,
                distance_matrix: np.ndarray,
                travel_time_matrix: np.ndarray,
                df_instances: pd.DataFrame,
                time_windows: list,
                capacity: float,
                big_M: float,
                penalty: float) -> tuple:
    r"""
    Construct the Gurobi model with all decision variables and constraints.

    Variables:
    - arc_used[i,j]: Binary, whether vehicle travels from i to j.
    - arrival[i]: Continuous, service start time at node i.
    - load[i]: Continuous, cumulative load after servicing i.
    - mtz[i]: Continuous, ordering index for subtour elimination.
    - served[i]: Binary, whether customer i is visited.

    Constraints:
    1. Routing continuity: in-degree = out-degree = served.
    2. Depot must have at least one departure and arrival.
    3. Time window adherence at each node.
    4. Vehicle capacity: load bounds.
    5. Time and load propagation using big-M conditioning.
    6. Subtour elimination via Miller–Tucker–Zemlin.

    :param num_nodes: Number of nodes.
    :param distance_matrix: Matrix of distances.
    :param travel_time_matrix: Matrix of travel times.
    :param df_instances: DataFrame of instance data.
    :param time_windows: List of (start, end) per node.
    :param capacity: Vehicle capacity.
    :param big_M: Big-M constant.
    :param penalty: Penalty for skipping customers.
    :return: Tuple (Model, arc_used variable dict).
    """
    model = Model('VRPTW')

    # Decision variables
    arc_used = model.addVars(num_nodes, num_nodes, vtype=GRB.BINARY, name='arc_used')
    arrival = model.addVars(num_nodes, vtype=GRB.CONTINUOUS, name='arrival_time')
    load = model.addVars(num_nodes, vtype=GRB.CONTINUOUS, name='load_after')
    mtz = model.addVars(num_nodes, vtype=GRB.CONTINUOUS, name='mtz_order')
    served = model.addVars(range(1, num_nodes), vtype=GRB.BINARY, name='served')

    # Objective: minimize travel distance + heavy penalty for each skipped customer
    model.setObjective(
        quicksum(distance_matrix[i][j] * arc_used[i,j]
                 for i in range(num_nodes) for j in range(num_nodes) if i != j)
        + penalty * quicksum(1 - served[i] for i in range(1, num_nodes)),
        GRB.MINIMIZE
    )

    # Routing constraints: ensure served nodes have exactly one in/out arc
    for i in range(1, num_nodes):
        model.addConstr(quicksum(arc_used[j,i] for j in range(num_nodes) if j!=i) == served[i])
        model.addConstr(quicksum(arc_used[i,j] for j in range(num_nodes) if j!=i) == served[i])

    # Depot constraints: at least one departure and return, no self-loop
    model.addConstr(quicksum(arc_used[0,j] for j in range(1,num_nodes)) >= 1)
    model.addConstr(quicksum(arc_used[i,0] for i in range(1,num_nodes)) >= 1)
    arc_used[0,0].ub = 0 # prevent trivial loop at depot

    # Time window constraints for each node
    for i in range(num_nodes):
        start, end = time_windows[i]
        model.addConstr(arrival[i] <= end) # must start service before window closes

    # Initialize depot time within day's horizon
    model.addConstr(arrival[0] >= 0)
    model.addConstr(arrival[0] <= big_M)

    # Capacity constraints: initial load zero, then accumulate demands
    model.addConstr(load[0] == 0)
    for i in range(1, num_nodes):
        model.addConstr(load[i] >= df_instances.loc[i,'demand'] * served[i])
        model.addConstr(load[i] <= capacity)

    # Time and load propagation: ensure consistency along arcs when used
    for i in range(1, num_nodes):
        for j in range(1, num_nodes):
            if i != j:
                # Enforce arrival time consistency if arc is used and both served
                model.addConstr(
                    arrival[j] >= arrival[i] + travel_time_matrix[i][j] + df_instances.loc[i,'service']
                    - big_M * (1 - arc_used[i,j] + 1 - served[i] + 1 - served[j])
                )
                # Enforce load accumulation under same conditions
                model.addConstr(
                    load[j] >= load[i] + df_instances.loc[j,'demand']
                    - capacity * (1 - arc_used[i,j] + 1 - served[i] + 1 - served[j])
                )

    # MTZ subtour elimination constraints
    for i in range(1, num_nodes):
        model.addConstr(mtz[i] >= 1)
        model.addConstr(mtz[i] <= num_nodes - 1)
    for i in range(1, num_nodes):
        for j in range(1, num_nodes):
            model.addConstr(
                mtz[i] - mtz[j] + (num_nodes - 1) * arc_used[i,j]
                <= num_nodes - 2 + num_nodes * (1 - served[i]) + num_nodes * (1 - served[j])
            )

    return model, arc_used


def solve_and_extract(model: Model, arc_used, distance_matrix: np.ndarray) -> None:
    r"""
    Execute optimization, handle infeasibility, and reconstruct routes from solution.

    :param model: Gurobi Model.
    :param arc_used: Gurobi tupledict of arc variables.
    :param distance_matrix: Matrix of distances for output.
    :return: None (prints results).
    """
    model.setParam('TimeLimit', 300) # cap solve time at 5 minutes
    model.optimize()

    if model.status in {GRB.OPTIMAL, GRB.TIME_LIMIT}:
        print(f"Total distance (with penalties): {model.objVal:.2f}")
        routes, visited = [], set()
        num_nodes = distance_matrix.shape[0]

        # Identify all routes by following arcs from depot
        for j in range(1, num_nodes):
            if arc_used[0,j].X > 0.5:
                route, length = [0], distance_matrix[0,j]
                curr = j
                # Follow the path until returning to depot
                while curr != 0:
                    route.append(curr)
                    for k in range(num_nodes):
                        if arc_used[curr,k].X > 0.5:
                            length += distance_matrix[curr,k]
                            curr = k
                            break
                    else:
                        # Dead-end: no outgoing arc found
                        break

                route.append(0)
                tpl = tuple(route)

                if tpl not in visited:
                    visited.add(tpl)
                    routes.append((route, length))

        print(f"Number of vehicles: {len(routes)}")
        for idx, (rt, ln) in enumerate(routes, 1):
            print(f"Vehicle {idx} route (Distance: {ln:.2f}): {rt}")

    elif model.status == GRB.INFEASIBLE:
        print("Model infeasible. Computing IIS...")
        model.computeIIS()
        model.write("model.ilp")  # IIS file shows minimal constraint set
        print("IIS written to model.ilp")
    else:
        print("No solution found.")

# Load and preprocess data
df_coords, df_demands, df_services, df_tw_raw = load_data()

# Merge into single DataFrame of instances
df_inst = merge_data(df_coords, df_demands, df_services)

# Compute distance and travel time matrices
dist_mat, tt_mat = compute_matrices(df_inst)

# Consolidate time windows into single interval per node
tw = consolidate_time_windows(df_tw_raw, df_inst['id'].to_numpy())

# Verify no single customer exceeds vehicle capacity
check_capacity(df_inst, 12600)

# Build optimization model with all constraints
model, arc_used = build_model(len(df_inst), dist_mat, tt_mat, df_inst, tw, capacity=12600, big_M=1440, penalty=1e6)

# Solve model and display routes
solve_and_extract(model, arc_used, dist_mat)