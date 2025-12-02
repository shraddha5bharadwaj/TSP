# TSP

## Run Commands

### Creating executable 
    cd code
    pyinstaller --onefile main.py
    mv dist/main exec
    chmod +x exec
### Running executable from root
    cd code
    ./exec -time 30 -seed 5 -alg Approx -inst '/Users/shrads/Desktop/Algos/Data/Atlanta.tsp'
    ./exec -time 30 -seed 5 -alg LS -inst '/Users/shrads/Desktop/Algos/Data/Atlanta.tsp'
    ./exec -time 30 -seed 5 -alg BF -inst '/Users/shrads/Desktop/Algos/Data/Atlanta.tsp'

### tests to check if main working

    Python Main.py -time 7 -seed 500 -alg BF  -inst /Users/shrads/Desktop/Algos/Data/UMissouri.tsp
    Python Main.py -time 7 -seed 500 -alg Approx  -inst /Users/shrads/Desktop/Algos/Data/UMissouri.tsp
    Python Main.py -time 7 -seed 500 -alg LS  -inst /Users/shrads/Desktop/Algos/Data/UMissouri.tsp
    Python Main.py -time 7  -alg Approx  -inst /Users/shrads/Desktop/Algos/Data/UMissouri.tsp 

### test to check if exec is working
    cd code
    ./exec -time 7 -seed 500 -alg BF -inst /Users/shrads/Desktop/Algos/Data/UMissouri.tsp
    ./exec -time 7 -seed 500 -alg Approx -inst /Users/shrads/Desktop/Algos/Data/UMissouri.tsp
    ./exec -time 7 -seed 500 -alg LS -inst /Users/shrads/Desktop/Algos/Data/UMissouri.tsp
    ./exec -time 7 -alg Approx -inst /Users/shrads/Desktop/Algos/Data/UMissouri.tsp

