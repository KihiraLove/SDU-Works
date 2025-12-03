# Robot Motion Planning: Design, Complexity, and Behavior

Comparing an **exact geometric method** vs. a **discretization-based planner**, with explicit complexity and accuracy trade-offs.

## 1. High-Level Structure

The program compares two approaches for planning a path from a start point to a goal point in a polygonal environment with obstacles:

* **Exact planner**: Visibility graph + Dijkstra
* **Approximate planner**: Uniform grid + BFS

Pipeline:

1. Read or construct a `Problem` (environment + start + goal)
2. Run both planners via `Runner`
3. Collect results in `PlanningResult`
4. Produce:

   * TikZ/LaTeX (+ optional PDF)
   * Optional Matplotlib PNG
   * Optional detailed numeric report

## 2. Complexity Summary

Let:

* `n_v` = number of obstacle vertices + start + goal
* `E_obs` = number of obstacle edges (≈ `n_v` for simple polygons)
* `G = grid_size²` = number of grid cells

### 2.1 Visibility Graph Planner

**Graph construction**

* Pairs of vertices: Θ(`n_v²`)
* For each pair, `segment_hits_obstacle` checks all obstacle edges: O(`E_obs`)
* Total complexity:
  **O(n_v² · E_obs)** ≈ **O(n_v³)** when `E_obs = Θ(n_v)`

Memory:

* Vertices: O(`n_v`)
* Adjacency list (possibly dense): up to O(`n_v²`) edges

**Shortest path (Dijkstra)**

* Complexity: **O((n_v + m_vg) log n_v)** with binary heap
* In a dense visibility graph (`m_vg = Θ(n_v²)`): **O(n_v² log n_v)**
* In practice, graph construction cost typically dominates for moderate sizes.

### 2.2 Grid-Based Planner

**Bounding square**

* Single pass over all vertices and start/goal: **O(n_v)**

**Cell classification**

* For each of `G` cells:
  * Call `point_in_any_obstacle` → O(`n_v`)
* Total: **O(G · n_v)**

**BFS on the grid**

* Nodes: `G` cells
* Edges: O(`G`) (each cell has at most 4 neighbors)
* BFS: **O(G)**

Memory:

* Grid: O(`G`)
* BFS structures (`visited`, `parent`): O(`G`)

In practice:

* For small `n_v` and large `grid_size`, the grid part dominates runtime.
* For large `n_v`, the visibility graph construction dominates.

## 3. Key Design Choices

### 3.1 Two Planners: Exact vs Approximate

* **Visibility graph**: Classical exact solution for polygonal shortest paths.

  * Good for theory and for reference.
  * Cubic worst-case complexity in this implementation.

* **Uniform grid + BFS**: Conceptually close to occupancy grids in robotics.

  * Simple to implement and reason about.
  * Quality governed by resolution (`grid_size`).
  * BFS guarantees minimal number of grid steps, not geometric shortest path.

### 3.2 Uniform Grid instead of Full Quadtree

* The grid corresponds to a full quadtree of fixed depth, but is stored explicitly as a 2D array of `GridCell`.
* This avoids quadtree implementation overhead while still allowing discussion of quadtrees in the report:
  * Leaves of a uniform quadtree can be indexed as `(i, j)` cells.
* Trade-off: no adaptive refinement; resolution is uniform everywhere.

### 3.3 Bounding Square and Margin

* The grid overlays a **square** (not a rectangle) that contains:
  * All obstacle vertices
  * Start and goal

* `margin_ratio` defines how much extra space is added around the environment:
  * Reduces clipping at boundaries.
  * Ensures that start/goal and obstacles are not too close to the grid border.

### 3.4 Discrete Free Space via Cell Centers

* A cell is considered **blocked** if its **center** is inside or on an obstacle.
* This is intentionally simple and compatible with standard grid-based planning:
  * The path is a polyline of cell centers.
  * Very thin obstacles or narrow passages may be misrepresented if the grid is too coarse.

### 3.5 Visibility Conditions

* `segment_hits_obstacle` **skips edges that share an endpoint** with either segment endpoint:
  * This allows visibility along polygon edges and between a vertex and its adjacent vertices.
  * Otherwise, segments that lie on polygon edges would be incorrectly flagged as blocked.

## 4. What Happens During a Run (Conceptual)

At a high level:

1. **Input stage**
   * Either parse a file with `START`, `GOAL`, and `OBSTACLE` blocks, or construct a built-in demo environment.
   * Validate start/goal (not equal, not inside obstacles).

2. **Exact planning stage**

   * Collect all obstacle vertices + start + goal as graph vertices.
   * For every pair:
     * Check if the straight segment between them intersects any obstacle edge (excluding shared endpoints).
     * If not, add a weighted edge to the visibility graph.
   
   * Run Dijkstra from start vertex to goal vertex.
   * If reachable, reconstruct the vertex sequence as the exact path.

3. **Grid planning stage**

   * Compute bounding square and derive `cell_size`.
   * For each grid cell:
     * Check if its center lies inside any obstacle.
     * Mark as free/blocked.
   
   * Map start and goal to grid cells.
   * If either mapped cell is blocked, the grid planner fails (for this resolution).
   * Otherwise run BFS on free cells to find a 4-connected path.
   * Convert the sequence of cells to a sequence of cell-center points.

4. **Output stage**

   * Draw:
     * Obstacles (gray polygons)
     * Visibility-graph path (blue polyline)
     * Grid path (red dashed polyline)
     * Start/goal markers
   
   * Optionally compile TikZ → PDF and optionally create a Matplotlib PNG.
   * Print a numeric report (lengths, number of vertices/steps, length ratio).

## 5. Special Cases and Failure Modes

### 5.1 Invalid Start/Goal

* If `start == goal`: abort with error.
* If start or goal lies inside any obstacle (including on its boundary): abort with error.
* These checks are applied both for file input and the demo environment.

### 5.2 No Path in Visibility Graph

* After Dijkstra:
  * If `dist[goal] = ∞`, the goal is unreachable in the visibility graph.

* Possible reasons:
  * Obstacles form a true barrier in continuous space.
  * Visibility graph is disconnected due to geometry.

The report explicitly notes when no visibility-graph path exists.

### 5.3 No Path on the Grid

Reasons the grid planner can fail:

1. **Start or goal maps to a blocked cell**:

   * At the chosen resolution, the cell center lies inside an obstacle.
   
2. **Grid disconnects free regions**:

   * Narrow passages may “disappear” at low resolution, even though a continuous path exists.

The report describes these reasons and suggests increasing `grid_size` if necessary.

### 5.4 Degenerate or Tiny Environments

* If all points have the same coordinates or there are too few points:

  * `_compute_bounding_square` enforces a minimum side length of 1.0.
  * This prevents division by zero and ensures a non-degenerate grid.

## 6. Geometric Tests: Subtle Points

### 6.1 Epsilon (`EPS`)

* All collinearity checks use an absolute tolerance `EPS = 1e-9`:
  * Reduce instability from rounding errors.
  * Points extremely close to collinear are treated as collinear.

This affects:
  * `orient`
  * `on_segment`
  * Intersection edge cases in `segments_intersect`

### 6.2 Point-on-Boundary is “Inside”

* `point_in_polygon` returns `True` if the point lies **on** any edge.
* Consequences:
  * Start/goal on an obstacle boundary are considered invalid.
  * Grid cells with centers exactly on edges are treated as blocked.

Avoids paths grazing obstacle boundaries in ambiguous ways.

### 6.3 Ray Casting: Half-Open Interval

In `point_in_polygon`:

* The edge is considered only if `p.y` lies in `[a.y, b.y)` (after ordering `a.y ≤ b.y`):
  * Prevents double counting intersections at vertices.
  * Ensures a stable parity rule for inside/outside classification.

### 6.4 Segment-Against-Obstacle Edges

* `segment_hits_obstacle` skips any obstacle edge that shares an endpoint with the tested segment:
  * This allows the visibility graph to include edges that follow polygon edges or connect adjacent vertices.

* For all other edges:
  * Full intersection logic (`segments_intersect`) is used:
    * Proper crossings, endpoint touches, and collinear overlaps all count as blocking.

### 6.5 Polyline Length

* Both paths (visibility and grid) are compared via `polyline_length`:
  * This uses Euclidean norm on consecutive pairs of points.
  * The “approx / exact length ratio” in the report is computed as:
    `length(grid_path) / length(vg_path)` if `vg_path` exists.

## 7. Results

* **Visibility path**:
  * Should be as short as possible given the polygonal obstacles.
  * Uses straight segments between obstacle vertices, start, and goal.

* **Grid path**:
  * Longer due to:
    * Manhattan-style stepping on the grid.
    * Discretization error from finite resolution.

* **Length ratio**:
  * Values close to 1 → high-quality approximation for this environment and `grid_size`.
  * Larger values → indicate either:
    * A coarse grid; or
    * Complex geometry poorly captured by uniform discretization.