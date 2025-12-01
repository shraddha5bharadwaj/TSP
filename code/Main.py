import sys
import os
import math
import random
import argparse
import os
import time
import random

from bf_solver import brute_force_tsp           
from mst_approx import mst_2approx_algorithm    
from local_search import local_search_tsp       



def parse_tsp_file(file_path):
    """
    Parse a TSPLIB-style .tsp file with NODE_COORD_SECTION and EUC_2D coords.
    Returns:
        coords: list of (x, y) floats, index i corresponds to vertex ID i+1.
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
    return math.hypot(dx, dy)


def tour_length(coords, tour):
    """
    Compute total length of a TSP tour.
    tour: list of vertex indices (0-based).
    Automatically closes the cycle (last -> first).
    """
    n = len(tour)
    total = 0.0
    for k in range(n):
        i = tour[k]
        j = tour[(k + 1) % n]
        total += euclid_dist(coords, i, j)
    return total





def run_and_write_solution(instance_path, method, cutoff, seed=None):
    # --- Load coordinates ---
    coords = parse_tsp_file(instance_path)

    # --- Choose algorithm ---
    method = method.upper()
    if method == "BF":
        tour_algo = brute_force_tsp
    elif method == "APPROX":
        tour_algo = mst_approx_algorithm
    elif method == "LS":
        if seed is None:
            raise ValueError("Local Search (LS) requires a seed")
        tour_algo = lambda coords: local_search_tsp(coords, cutoff, seed)
    else:
        raise ValueError(f"Unknown algorithm: {method}")

    # --- Set RNG seed if used ---
    if seed is not None:
        random.seed(seed)

    # --- Run + time ---
    start = time.time()
    tour = tour_algo(coords)
    runtime = time.time() - start

    # --- Compute cost ---
    cost = tour_length(coords, tour)

    # --- Build .sol filename ---
    instance_name = os.path.splitext(os.path.basename(instance_path))[0].lower()

    if method == "BF":
        sol_name = f"{instance_name} BF {cutoff}.sol"
    elif method == "APPROX":
        sol_name = f"{instance_name} Approx.sol" if seed is None else f"{instance_name} Approx {seed}.sol"
    else:   # LS
        sol_name = f"{instance_name} LS {cutoff} {seed}.sol"

    # --- Format output lines ---
    line1 = f"{cost:.6f}"
    line2 = ",".join(str(i + 1) for i in tour)

    # --- Print to stdout ---
    print(line1)
    print(line2)

    # --- Write solution file ---
    with open(sol_name, "w") as f:
        f.write(line1 + "\n")
        f.write(line2 + "\n")

    return runtime


def main():
    parser = argparse.ArgumentParser(description="TSP Solver Front-End")

    parser.add_argument("-inst", required=True, help="Path to input .tsp file")
    parser.add_argument("-alg", required=True, choices=["BF", "Approx", "LS"], help="Algorithm")
    parser.add_argument("-time", required=True, type=int, help="Cutoff time in seconds")
    parser.add_argument("-seed", type=int, help="Random seed (only for LS or if Approx uses randomness)")

    args = parser.parse_args()

    run_and_write_solution(
        instance_path=args.inst,
        method=args.alg,
        cutoff=args.time,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
