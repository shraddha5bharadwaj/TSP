import math
import sys
import time
import os


def parse_tsp_file(file_path):
    """
    Parse a TSPLIB-style .tsp file with NODE_COORD_SECTION and EUC_2D coords.
    Returns:
        coords: list of (x, y) floats, where index i corresponds to vertex ID i+1.
    """
    coords = []
    in_coord_section = False

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("NODE_COORD_SECTION"):
                in_coord_section = True
                continue

            if not in_coord_section:
                # ignore header lines like NAME, DIMENSION, etc.
                continue

            if line.startswith("EOF"):
                break

            parts = line.split()
            if len(parts) < 3:
                continue

            vid = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])

            # ensure coords list is big enough
            if len(coords) < vid:
                coords.extend([None] * (vid - len(coords)))

            coords[vid - 1] = (x, y)

    return coords


# ---------- Geometry / tour length ----------

def euclid_dist(coords, i, j):
    """
    Euclidean distance between vertex i and j (0-based indices).
    """
    x1, y1 = coords[i]
    x2, y2 = coords[j]
    dx = x1 - x2
    dy = y1 - y2
    return math.hypot(dx, dy)  # sqrt(dx*dx + dy*dy)


def tour_length(coords, tour):
    """
    Compute total length of a TSP tour.
    tour: list of vertex indices (0-based), e.g. [0, 5, 2, ..., 7].
    Automatically closes the cycle (last -> first).
    """
    n = len(tour)
    total = 0.0
    for k in range(n):
        i = tour[k]
        j = tour[(k + 1) % n]  # wrap around
        total += euclid_dist(coords, i, j)
    return total


# ---------- 2-approx TSP via MST (metric TSP) ----------

def mst_approx_algorithm(coords):
    """
    2-approximation algorithm for metric TSP using MST

    Steps -
      1. Prim's algorithm we compute MST of the complete graph given the coordiantes with euclidean distance as weights.
      2. Do a DFS preorder traversal of the MST to get a Hamiltonian cycle - visit nodes in order of first visit
      3. Returns - total euclidean cost and the list of nodes in which it was visited in  the TSP tour.

    Input - 
      coords- list of (x, y) coordinates, index i == vertex ID i+1

    Output -
      tour- list of 0-based vertex indices in visiting order.
    """
    n = len(coords)
    if n == 0:
        return []

    # --- Step 1: Prim's algorithm on implicit complete graph ---

    in_mst = [False] * n
    key = [float("inf")] * n   # best edge weight to connect each vertex
    parent = [-1] * n          # parent[v] = u in MST

    # Start at vertex 0 (arbitrary)
    key[0] = 0.0

    for _ in range(n):
        # pick vertex u not in MST with smallest key[u]
        u = -1
        u_key = float("inf")
        for v in range(n):
            if not in_mst[v] and key[v] < u_key:
                u_key = key[v]
                u = v

        in_mst[u] = True

        # update keys of neighbors (all other vertices) since graph is complete
        for v in range(n):
            if not in_mst[v]:
                w = euclid_dist(coords, u, v)
                if w < key[v]:
                    key[v] = w
                    parent[v] = u

    # --- Build adjacency list of MST ---
    adj = [[] for _ in range(n)]
    for v in range(1, n):
        p = parent[v]
        if p != -1:
            adj[p].append(v)
            adj[v].append(p)

    # --- Step 2: DFS preorder to get tour ---

    tour = []
    visited = [False] * n

    def dfs(u):
        visited[u] = True
        tour.append(u)
        for w in adj[u]:
            if not visited[w]:
                dfs(w)

    dfs(0)  # root at 0; any start vertex works for metric TSP

    # tour is now a permutation of [0..n-1] in preorder
    return tour


# ---------- Driver ----------



def run_tsp(file_path,cutoff=0,seed=None):
    """
    Run a TSP algorithm, measure runtime, print results, and write .sol file.

    Parameters:
        file_path:     path to .tsp file
        algorithm:     function(coords) -> list of 0-based vertex indices
        sol_filename:  name of output .sol file (e.g., "atlanta Approx 600.sol")

    Prints:
        line 1: total cost (float)
        line 2: comma-separated vertex IDs (1-based)

    Returns:
        runtime (seconds)
    """
    

    if file_path is None or not isinstance(file_path, str) or not os.path.isfile(file_path):
        raise ValueError(f"Invalid file path: {file_path}")

    if seed is None:
        sol_filename = "../output/approximate/" + os.path.splitext(os.path.basename(file_path))[0] + "_Approx"+".sol"
    else:
        sol_filename = "../output/approximate/" + os.path.splitext(os.path.basename(file_path))[0] + "_Approx_"+str(seed)+".sol"

    # ---- Parse TSP ----
    coords = parse_tsp_file(file_path)

    # ---- Run Algorithm + time it ----
    start = time.time()
    tour = mst_approx_algorithm(coords)
    end = time.time()
    runtime = end - start

    # ---- Compute solution quality ----
    best_cost = tour_length(coords, tour)

    # ---- Prepare output strings ----
    line1 = f"{best_cost:.6f}"
    line2 = ",".join(str(i + 1) for i in tour)

    # ---- Print to stdout ----
    print(line1)
    print(line2)

    # ---- Write to .sol file ----
    with open(sol_filename, "w") as f:
        f.write(line1 + "\n")
        f.write(line2 + "\n")
    print(repr(runtime))
    return runtime


# Takes in the filename as a an argument to run the mst_2approx_algorithm

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tsp_mst.py <instance.tsp>")
        sys.exit(1)

    file_path = sys.argv[1]
    a = run_tsp(file_path)
    print(f"{a} seconds")
