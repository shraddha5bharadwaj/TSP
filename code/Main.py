import sys
import os
import math
import random
import argparse
import os
import time
import random

### Takes in the algorithm choice, instance file path, cutoff time, and optional seed and produce output file in respective folder in output
# from bf_solver import brute_force_tsp           
from approximation import run_tsp_approx    
# from local_search import local_search_tsp       
 
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




# def euclid_dist(coords, i, j):
#     """
#     Euclidean distance between vertex i and j (0-based indices).
#     """
#     x1, y1 = coords[i]
#     x2, y2 = coords[j]
#     dx = x1 - x2
#     dy = y1 - y2
#     return math.hypot(dx, dy)


# def tour_length(coords, tour):
#     """
#     Compute total length of a TSP tour.
#     tour: list of vertex indices (0-based).
#     Automatically closes the cycle (last -> first).
#     """
#     n = len(tour)
#     total = 0.0
#     for k in range(n):
#         i = tour[k]
#         j = tour[(k + 1) % n]
#         total += euclid_dist(coords, i, j)
#     return total




def run_and_write_solution(instance_path, method, cutoff, seed=None):
    # --- Load coordinates ---
    coords = parse_tsp_file(instance_path)

    # --- Choose algorithm ---
    start = time.time()
    method = method.upper()
    if method == "BF":
        tour_algo = brute_force_tsp
    elif method == "APPROX":
        tour_algo = run_tsp_approx(file_path=instance_path,cutoff=cutoff,seed=seed)
    elif method == "LS":
        if seed is None:
            raise ValueError("Local Search (LS) requires a seed")
        tour_algo = local_search_tsp(file_path = instance_path, coords = coords, cutoff = cutoff, seed = seed)
    else:
        raise ValueError(f"Unknown algorithm: {method}")
    runtime = time.time() - start
    # --- Set RNG seed if used ---
    if seed is not None:
        random.seed(seed)


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
