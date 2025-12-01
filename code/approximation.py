import math
import sys
import time
import os


def read_tsp_file(file_path):
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




def euclid_dist(coords, i, j):
    """
    Euclidean distance between vertex i and j
    """
    x1, y1 = coords[i]
    x2, y2 = coords[j]
    dx = x1 - x2
    dy = y1 - y2
    return math.hypot(dx, dy)  # sqrt(dx*dx + dy*dy)


def tour_length(coords, tour):
    """
    Computes total length of a TSP tour which is  tour cost and solution quality
    tour: list of vertex indices 
    Considers the last traversal from last to first vertex in the cost
    """
    n = len(tour)
    total = 0.0
    for k in range(n):
        i = tour[k]
        j = tour[(k + 1) % n]  # wrap around
        total += euclid_dist(coords, i, j)
    return total


# 2-approx TSP through MST

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

    #  Prim's algorithm to build MST

    mstArray = [False] * n
    key = [float("inf")] * n   # best edge weight to connect each vertex
    parent = [-1] * n          # parent[v] = u in MST

    # Start at vertex 0 (arbitrary)
    key[0] = 0.0

    for _ in range(n):
        # pick vertex u not in MST with smallest key[u]
        u = -1
        u_key = float("inf")
        for v in range(n):
            if not mstArray[v] and key[v] < u_key:
                u_key = key[v]
                u = v

        mstArray[u] = True

        # update keys of neighbors (all other vertices) since graph is complete
        for v in range(n):
            if not mstArray[v]:
                w = euclid_dist(coords, u, v)
                if w < key[v]:
                    key[v] = w
                    parent[v] = u

    # Graph adjacency list based MST graph construction
    Graph = [[] for _ in range(n)]
    for node in range(1, n):
        p = parent[node]
        if p != -1:
            Graph[p].append(node)
            Graph[node].append(p)

    #  DFS preorder algorithm function to get tour 

    tour = []
    visited = [False] * n

    def dfs(u):
        visited[u] = True
        tour.append(u)
        for w in Graph[u]:
            if not visited[w]:
                dfs(w)

    dfs(0)  # root at 0, preorder traversal

    # tour is now a permutation of [0..n-1] in preorder
    return tour



def run_tsp_approx(file_path,cutoff=0,seed=None):
    """
    This function does the following
        Runs a TSP algorithm, measures runtime, prints results, and writes .sol file.

    Parameters:
        file_path   path to .tsp file
        cutoff      time cutoff in seconds - not used
        seed        random seed - not used

    Prints:
        line 1: total cost , solution quality (float)
        line 2: comma-separated vertex IDs starts from 1

    Returns:
        runtime (seconds)
    """
    
    # check file path validity
    if file_path is None or not isinstance(file_path, str) or not os.path.isfile(file_path):
        raise ValueError(f"Invalid file path: {file_path}")

    if seed is None:
        sol_filename = "../output/approximate/" + os.path.splitext(os.path.basename(file_path))[0] + "_Approx"+".sol"
    else:
        sol_filename = "../output/approximate/" + os.path.splitext(os.path.basename(file_path))[0] + "_Approx_"+str(seed)+".sol"

    # read tsp and get all vertex coordinates
    coords = read_tsp_file(file_path)

    # run the mst approx algorithm and time it
    start = time.time()
    tour = mst_approx_algorithm(coords)
    end = time.time()
    runtime = end - start

    # computing cost of the returned tour
    best_cost = tour_length(coords, tour)

    #Preparing output
    line1 = f"{best_cost:.6f}"
    line2 = ",".join(str(i + 1) for i in tour)

    
    print(line1)
    print(line2)

    # Write to .sol file
    with open(sol_filename, "w") as f:
        f.write(line1 + "\n")
        f.write(line2 + "\n")
    print(repr(runtime))
    return runtime


# for indivisual testing
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tsp_mst.py <instance.tsp>")
        sys.exit(1)

    file_path = sys.argv[1]
    a = run_tsp_approx(file_path)
    print(f"{a} seconds")
