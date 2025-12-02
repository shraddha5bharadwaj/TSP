# TSP

## Run Commands

### Creating executable 
- pyinstaller --onefile main.py
- mv dist/main exec
- chmod +x exec
### Running executable from root
    cd code
    ./exec -time 30 -seed 5 -alg Approx -inst '/Users/shrads/Desktop/Algos/Data/Atlanta.tsp'
    ./exec -time 30 -seed 5 -alg LS -inst '/Users/shrads/Desktop/Algos/Data/Atlanta.tsp'
    ./exec -time 30 -seed 5 -alg BF -inst '/Users/shrads/Desktop/Algos/Data/Atlanta.tsp'
