import os
import numpy as np
import pandas as pd

def collect_data(systems, algorithms, runs, base_path='../results'):

    results_list = []

    for system in systems:
        system_path = os.path.join(base_path, system, 'tuning_results')
        env_path = os.path.join(system_path, algorithms[0], 'optimized_pop_perf_run_0')
        environments = sorted([f.replace('_perf.csv', '') for f in os.listdir(env_path) if f.endswith('_perf.csv')])

        for env in environments:

            result = {'Environment': env}
            algo_data = {}
            for algo in algorithms:
                algo_time = []
                for run in range(runs):
                    file_path = os.path.join(system_path, algo, f'optimized_pop_perf_run_{run}', f'{env}_time.csv')
                    if os.path.exists(file_path):
                        run_df = pd.read_csv(file_path, header=None, nrows=1)
                        algo_time.extend(run_df.iloc[:, 0].tolist())

                result[f'{algo}_time'] = round(np.mean(algo_time), 4)

            results_list.append(result)


    results_df = pd.DataFrame(results_list)
    results_df.to_csv(f'RQ2/Running_Time.csv', index=False)

def main():

    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    compared_algorithms = ['FEMOSAA', 'Seed-EA', 'D-SOGA', 'LiDOS', 'DLiSA-0.3']
    runs = 100
    collect_data(systems, compared_algorithms, runs)


if __name__ == '__main__':
    main()


