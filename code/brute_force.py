"""
This file implements a brute-force solver for the Traveling Salesperson Problem (TSP), designed to find the optimal path 
visiting all nodes. It handles file parsing, Euclidean distance calculations, and output generation within a specified time limit.
"""
import sys
import math
import time
import argparse
import itertools
import os

class TSPSolver:
    def __init__(self, filename, cutoff_time):
        # Initializes the solver instance with the target filename and maximum allowed execution time.
        # Sets up the initial state, including empty node lists, infinite best cost, and timing variables.
        self.filename = filename
        self.cutoff_time = cutoff_time
        self.name = "Unknown"
        self.dimension = 0
        self.nodes = []
        self.best_cost = float('inf')
        self.best_path = []
        self.start_time = 0
        self.full_tour = False 

    def parse_file(self):
        # Reads the input file to extract graph metadata (name, dimension) and coordinate data.
        # Iterates through lines to identify the coordinate section and populates the self.nodes list.
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        coord_section = False
        for line in lines:
            line = line.strip()
            if line.startswith("NAME"):
                self.name = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"):
                self.dimension = int(line.split(":")[1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                coord_section = True
                continue
            elif line == "EOF":
                break
            elif coord_section:
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    self.nodes.append({'id': node_id, 'x': x, 'y': y})

    def calc_distance(self, n1, n2):
        # Calculates the Euclidean distance between two node dictionaries using their 'x' and 'y' coordinates.
        # Returns the distance as an integer rounded to the nearest whole number (standard TSP convention).
        dx = n1['x'] - n2['x']
        dy = n1['y'] - n2['y']
        dist = math.sqrt(dx**2 + dy**2)
        return int(dist + 0.5)

    def solve_brute_force(self):
        # Explores all possible permutations of nodes (excluding the start) to find the path with the lowest cost.
        # Checks the elapsed time against the cutoff_time periodically to stop execution if the limit is exceeded.
        self.start_time = time.time()
        
        if not self.nodes:
            return

        initial_path = self.nodes[:]
        initial_cost = 0
        for i in range(len(initial_path) - 1):
            initial_cost += self.calc_distance(initial_path[i], initial_path[i+1])
        initial_cost += self.calc_distance(initial_path[-1], initial_path[0])
        
        self.best_cost = initial_cost
        self.best_path = initial_path

        start_node = self.nodes[0]
        other_nodes = self.nodes[1:]
        
        check_interval = 1000
        count = 0
        
        for p in itertools.permutations(other_nodes):
            count += 1
            
            if count % check_interval == 0:
                if time.time() - self.start_time > self.cutoff_time:
                    break

            current_path = [start_node] + list(p)
            current_cost = 0
            valid_path = True
            
            for i in range(len(current_path) - 1):
                current_cost += self.calc_distance(current_path[i], current_path[i+1])
                
                if current_cost >= self.best_cost:
                    valid_path = False
                    break
            
            if valid_path:
                current_cost += self.calc_distance(current_path[-1], start_node)
                
                if current_cost < self.best_cost:
                    self.best_cost = current_cost
                    self.best_path = current_path

        self.full_tour = (len(self.best_path) == self.dimension)

    def write_output(self, method_code):
        # Constructs the output filename and directory, then writes the solution quality and path to a .sol file.
        # Formats the output to include the total cost, the ordered sequence of node IDs, and the tour status.
        subfolder = "brute_force"
        os.makedirs(subfolder, exist_ok=True)

        cutoff_str = str(int(self.cutoff_time))
        filename_base = self.name

        output_filename = os.path.join(
            subfolder,
            f"{filename_base}_{method_code}_{cutoff_str}.sol"
        )

        quality = float(self.best_cost)
        path_ids = [str(n['id']) for n in self.best_path]
        path_str = ",".join(path_ids)
        full_tour_str = "Yes" if self.full_tour else "No"

        try:
            with open(output_filename, 'w') as f:
                f.write(f"{quality}\n")
                f.write(path_str + "\n")
                f.write(f"Full Tour: {full_tour_str}\n")
            print(f"Solution written to: {output_filename}")
        except IOError as e:
            print(f"Error writing output file: {e}")