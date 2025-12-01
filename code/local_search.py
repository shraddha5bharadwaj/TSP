import math
import random
import time
import argparse
import os
from typing import List, Tuple

Point = Tuple[float, float]
Tour = List[int]

def euclidean_dist(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def total_length(tour: Tour, points: List[Point]) -> float:
    """Total closed tour length (sum of edge lengths, closing to the start)."""
    L = 0.0
    n = len(tour)
    for i in range(n):
        a = points[tour[i]]
        b = points[tour[(i + 1) % n]]
        L += euclidean_dist(a, b)
    return L

# -------------------------
# Heuristics
# -------------------------
def nearest_neighbor(points: List[Point], start: int = 0) -> Tour:
    """Greedy nearest neighbor construction (O(n^2))."""
    n = len(points)
    if n == 0:
        return []
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    while unvisited:
        last = tour[-1]
        # choose nearest unvisited
        nxt = min(unvisited, key=lambda j: euclidean_dist(points[last], points[j]))
        tour.append(nxt)
        unvisited.remove(nxt)
    return tour

def coord_to_Point(coords):
    coords_point = list(range(len(coords)))
    for i in range(len(coords)):
        x: Point = [coords[i][0], coords[i][1]]
        coords_point[i] = x
    return coords_point

def two_opt(points: List[Point], tour: Tour = None) -> Tour:
    """
    2-opt local search: repeatedly reverse tour segments to improve length.
    This is a simple O(n^2) improvement per pass; runs until no improvement.
    Returns a (locally) improved tour.
    """
    n = len(points)
    if n <= 2:
        return list(range(n)) if tour is None else tour.copy()
    if tour is None:
        tour = list(range(n))
    else:
        tour = tour.copy()

    improved = True
    # iterate until no improvements
    while improved:
        improved = False
        # consider all pairs (i, j) with i < j, skip trivial adjacent edges
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Don't consider adjacent edges (j == i+1) because reversing would be no-op for cycle connectivity check
                if j == i + 1:
                    continue
                a, b = tour[i - 1], tour[i]
                c, d = tour[j - 1], tour[j % n]
                # current distance of (a-b) + (c-d)
                current = euclidean_dist(points[a], points[b]) + euclidean_dist(points[c], points[d])
                # after 2-opt, edges become (a-c) + (b-d)
                new = euclidean_dist(points[a], points[c]) + euclidean_dist(points[b], points[d])
                if new + 1e-12 < current:
                    # perform reversal of segment [i:j-1] (inclusive)
                    tour[i:j] = reversed(tour[i:j])
                    improved = True
        # loop until no improvement found in full scan
    return tour

# -------------------------
# Simulated Annealing
# -------------------------
def simulated_annealing(
    points: List[Point],
    init_tour: Tour = None,
    max_iters: int = 10000,
    T0: float = None,
    alpha: float = 0.995,
    move: str = '2opt',
    seed: int = None
) -> Tuple[Tour, float, dict]:
    """
    Simulated annealing for TSP.

    Params:
      - points: list of 2D points
      - init_tour: starting permutation (if None, a random tour is used)
      - max_iters: number of iterations (proposal attempts)
      - T0: initial temperature (if None, heuristically set)
      - alpha: multiplicative cooling factor per iteration (0 < alpha < 1)
      - move: '2opt' or 'swap' move proposal
      - seed: optional random seed for reproducibility

    Returns:
      - best tour found, its length, and stats dict (final temperature, iterations, times)
    Notes:
      - This implementation recalculates full tour length on proposals for simplicity.
        For speed, one can implement incremental delta updates for both swap and 2-opt.
    """
    if seed is not None:
        random.seed(seed)

    n = len(points)
    if n == 0:
        return [], 0.0, {'iters': 0, 'time': 0.0, 'final_T': None}

    # initial tour
    if init_tour is None:
        tour = list(range(n))
        random.shuffle(tour)
    else:
        tour = init_tour.copy()

    best = tour.copy()
    best_len = total_length(best, points)
    cur_tour = tour.copy()
    cur_len = best_len

    # initial temperature heuristic: fraction of average edge length times n
    if T0 is None:
        avg_edge = cur_len / n
        T = avg_edge * 0.5   # heuristic scaling; user can override
    else:
        T = T0

    start_time = time.time()
    it = 0
    while it < max_iters:
        it += 1

        if move == 'swap':
            # pick two positions and swap them
            i, j = sorted(random.sample(range(n), 2))
            new_tour = cur_tour.copy()
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        elif move == '2opt':
            # pick two cut points and reverse segment between them
            i, j = sorted(random.sample(range(n), 2))
            if i == j:
                continue
            # inclusive slice reversal
            new_tour = cur_tour.copy()
            # reverse subsegment from i to j (inclusive)
            new_tour[i:j+1] = reversed(new_tour[i:j+1])
        else:
            raise ValueError("Unknown move type: choose 'swap' or '2opt'")

        new_len = total_length(new_tour, points)
        delta = new_len - cur_len

        # accept if better or with Boltzmann probability
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
            cur_tour = new_tour
            cur_len = new_len
            if cur_len < best_len:
                best_len = cur_len
                best = cur_tour.copy()

        # cooling schedule
        T *= alpha

    elapsed = time.time() - start_time
    stats = {'iters': it, 'time': elapsed, 'final_T': T}
    return best, best_len, stats


def local_search_tsp(file_path, coords, cutoff, seed):
    points = coord_to_Point(coords)
    #print(f"Generated random instance with n={args.n}, seed={args.seed}")

    # Nearest Neighbor
    t0 = time.time()
    nn_tour = nearest_neighbor(points, start=0)
    nn_time = time.time() - t0
    #print_tour_info("Nearest Neighbor", nn_tour, points, nn_time) 
    # 2-opt starting from NN  
    t0 = time.time()
    two_tour = two_opt(points, tour=nn_tour)
    two_time = time.time() - t0
    #print_tour_info("2-opt (from NN)", two_tour, points, two_time)

    sa_time = 0
    nn_len = 0
    two_len = 0
    avg_sa_len = 0
    # Base RNG to produce per-run seeds and reproducible shuffles
    base_rand = random.Random(seed) if seed is not None else random.Random()
    best_sa_len = 0
    best_sa_time = 0
    # Run for 10 times with distinct random seeds
    for i in range(10):
        # Random initial tour (use base_rand for reproducibility when seed provided)
        init_random = list(range(len(points)))
        base_rand.shuffle(init_random)
        # generate a fresh seed for this SA run
        run_seed = base_rand.randint(0, 2**31 - 1)
        sa_tour, sa_len, sa_stats = simulated_annealing(
            points,
            init_tour=two_tour,            # start from two-opt result (common practice)
            seed=run_seed
        )
        sa_time += sa_stats['time']
        if sa_len < best_sa_len and i > 0:
            best_sa_len = sa_len
            best_sa_time = sa_stats['time']
            #best_sa_tour = sa_tour
        elif i == 0:
            best_sa_len = sa_len
            best_sa_time = sa_stats['time']
            #best_sa_tour = sa_tour
        avg_sa_len += sa_len#total_length(sa_tour, points)
        nn_len += total_length(nn_tour, points)
        two_len += total_length(two_tour, points)

    sa_time = sa_time / 10
    avg_sa_len = avg_sa_len / 10
    nn_len = nn_len / 10
    two_len = two_len /10
    print("Best SA length: " + str(best_sa_len) + "\n")
    print("Best SA time: " + str(best_sa_time) + "\n")
    #def run_tsp(file_path,cutoff=0,seed=None):
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
    

    #if file_path is None or not isinstance(file_path, str) or not os.path.isfile(file_path):
    #    raise ValueError(f"Invalid file path: {file_path}")

    sol_filename = "../output/local_search/" + os.path.splitext(os.path.basename(file_path))[0] + "_LS_"+str(cutoff) + "_" +str(seed)+".sol"

    # ---- Parse TSP ----
    #coords = parse_tsp_file(file_path)

    # ---- Run Algorithm + time it ----
    #start = time.time()
    #tour = local_search_tsp(coords)
    #end = time.time()
    runtime = two_time + sa_time + nn_time

    # ---- Compute solution quality ----
    #best_cost = tour_length(coords, tour)
    best_cost = avg_sa_len

    # ---- Prepare output strings ----
    line1 = f"{best_cost:.6f}"
    line2 = ",".join(str(i + 1) for i in sa_tour)

    # ---- Print to stdout ----
    print(line1)
    print(line2)

    # ---- Write to .sol file ----
    with open(sol_filename, "w") as f:
        f.write(line1 + "\n")
        f.write(line2 + "\n")
    print(repr(runtime))
    return runtime
