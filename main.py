import random
import os
import json
from Adaptation_Optimizer import AdaptationOptimizer

def main():

    '''
    +------------------------------Parameter Settings-------------------------------+
    + run: the number of independently run                                          +
    + max_generation: maximum generation of evolution in each workload environment  +
    + pop_size: population size                                                     +
    + mutation_rate: the probability of mutation                                    +
    + crossover_rate: the probability of crossover                                  +
    + systems: the corresponding system that algorithm is going to tuning           +
    + compared_algorithm: selected algorithm for tuning systems                     +
    + ------------------------------------------------------------------------------+

    remark: the optimized objective is executed time except h2 system (where the objective is throughput)
    '''

    run = 100
    max_generation = 3
    pop_size = 20
    mutation_rate = 0.1
    crossover_rate = 0.9
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    compared_algorithms = ['FEMOSAA', 'Seed-EA', 'D-SOGA', 'LiDOS', 'DLiSA-0.3']

    for i in range(run):
        for system in systems:
            if system == 'h2':
                optimization_goal = 'maximum'
            else:
                optimization_goal = 'minimum'

            data_folder = f'dataset/{system}'
            order_file = f'order_files/order_{system}_{i}.json'

            if os.path.exists(order_file):
                with open(order_file, 'r') as f:
                    data_files = json.load(f)
            else:
                data_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])
                random.shuffle(data_files)  # Randomize the order of workloads
                # save the workload sequence for each run
                os.makedirs(os.path.dirname(order_file), exist_ok=True)
                with open(order_file, 'w') as f:
                    json.dump(data_files, f)

            optimizer = AdaptationOptimizer(max_generation, pop_size, mutation_rate, crossover_rate, compared_algorithms, system, optimization_goal)
            optimizer.dynamic_optimization(data_folder, data_files, i)


if __name__== "__main__":
    main()