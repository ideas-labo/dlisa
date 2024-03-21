import os

import numpy as np
from rpy2.robjects import r, pandas2ri
import pandas as pd
from rpy2.robjects.packages import importr
from collections import Counter


pandas2ri.activate()
devtools = importr("devtools")
devtools.install_github("klainfo/ScottKnottESD", ref="development")
sk = importr('ScottKnottESD')

def count_rank_ones(results_df):

    rank_ones_df = results_df[results_df['r'] == 1]
    rank_ones_count = Counter(rank_ones_df['Algorithm'])
    return rank_ones_count

def calculate_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    return iqr

def collect_data(systems, algorithms, runs, base_path='../../results'):

    all_result_list = []

    for system in systems:
        system_path = os.path.join(base_path, system, 'tuning_results')
        env_path = os.path.join(system_path, algorithms[0], 'optimized_pop_perf_run_0')
        environments = sorted([f.replace('_perf.csv', '') for f in os.listdir(env_path) if f.endswith('_perf.csv')])

        results_list = []
        for env in environments:

            env_global_min, env_global_max = float('inf'), float('-inf')


            algo_data = {}
            for algo in algorithms:
                algo_values = []
                for run in range(runs):
                    file_path = os.path.join(system_path, algo, f'optimized_pop_perf_run_{run}', f'{env}_perf.csv')
                    if os.path.exists(file_path):
                        run_df = pd.read_csv(file_path, header=None, nrows=1)
                        algo_values.extend(run_df.iloc[:, 0].tolist())

                        env_global_min = min(env_global_min, min(algo_values))
                        env_global_max = max(env_global_max, max(algo_values))

                algo_data[algo] = algo_values



            algo_data_dataframe = pd.DataFrame(algo_data)
            r_sk = sk.sk_esd(algo_data_dataframe, version='p')
            column_order = [i - 1 for i in r_sk[3]]



            if system != 'h2':
                column_order = [i - 1 for i in r_sk[3]][::-1]


            ranking_results = pd.DataFrame(
                {
                    "Algorithms": [algo_data_dataframe.columns[i] for i in column_order],
                    "rankings": r_sk[1].astype("int"),
                }
            ).set_index("Algorithms")

            # Add the rankings to the results_list
            for algo in algorithms:
                algo_values = algo_data[algo]
                mean = round(np.mean(algo_values), 3)
                std = round(np.std(algo_values), 3)
                iqr = round(calculate_iqr(algo_values), 3)
                rank = ranking_results.loc[algo, 'rankings']


                norm_25 = round((np.percentile(algo_values, 25) - env_global_min) / (env_global_max - env_global_min) * 100,
                                4)
                norm_IQR = round((np.percentile(algo_values, 50) - env_global_min) / (env_global_max - env_global_min) * 100,
                                4)
                norm_75 = round((np.percentile(algo_values, 75) - env_global_min) / (env_global_max - env_global_min) * 100,
                                4)


                # Construct the result dictionary
                result = {
                    'Workload': env,
                    'Algorithm': algo,
                    'r': rank,
                    'Mean (Std)': f'{mean} ({std})'
                    #'Med_Visualization': '\quart{' + str(norm_25) + '}{' + str(norm_50) + '}{' + str(norm_75) + '}{100}'

                }

                results_list.append(result)
                all_result_list.append(result)

        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f'../RQ4/{system}.csv', index=False)

    all_results_df =pd.DataFrame(all_result_list)
    all_results_df.to_csv(f'../RQ4/scott_knott.csv')

def main():

    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    compared_algorithms = ['DLiSA-0.0', 'DLiSA-0.1', 'DLiSA-0.2', 'DLiSA-0.3', 'DLiSA-0.4', 'DLiSA-0.5', 'DLiSA-0.6', 'DLiSA-0.7', 'DLiSA-0.8', 'DLiSA-0.9']
    runs = 100
    collect_data(systems, compared_algorithms, runs)

    results_df = pd.read_csv(f'../RQ4/scott_knott.csv')
    rank_ones_count = count_rank_ones(results_df)

    for method, count in rank_ones_count.items():
        print(f"{method} ranked first {count} times.")



if __name__ == '__main__':
    main()


