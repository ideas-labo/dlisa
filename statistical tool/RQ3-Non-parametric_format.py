import os
import pandas as pd
from scipy import stats
from scipy.stats import ranksums, rankdata
import numpy as np


def calculate_iqr(data):

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    return iqr

# A12
def calculate_a12(group1, group2):
    n1, n2 = len(group1), len(group2)
    ranked = rankdata(np.concatenate((group1, group2)))
    rank1 = ranked[:n1]
    rank2 = ranked[n1:]
    a12 = ((rank1.sum() - (n1 * (n1 + 1)) / 2) / (n1 * n2))
    return a12 if a12 > 0.5 else (1 - a12)


'''
# matlab version -> python version (the results are the same to above)
def calculate_a12_test(x, y):
   
    stat, p = ranksums(x, y)
    rank_sum = np.sum(rankdata(np.concatenate((x, y)))[:len(x)])
    a = (rank_sum / len(x) - (len(x)+1) / 2) / len(y)
    return a
'''

def collect_data(systems, algorithms, runs, base_path='../results'):


    for system in systems:

        results_list = []
        system_path = os.path.join(base_path, system, 'tuning_results')
        env_path = os.path.join(system_path, algorithms[0], 'optimized_pop_perf_run_0')
        environments = sorted([f.replace('_perf.csv', '') for f in os.listdir(env_path) if f.endswith('_perf.csv')])

        for env in environments:
            # Initialize the global min and max under current env
            env_global_min, env_global_max = float('inf'), float('-inf')

            # collect the data of each algorithm
            algo_data = {}
            for algo in algorithms:
                algo_values = []
                for run in range(runs):
                    file_path = os.path.join(system_path, algo, f'optimized_pop_perf_run_{run}', f'{env}_perf.csv')
                    if os.path.exists(file_path):
                        run_df = pd.read_csv(file_path, header=None, nrows=1)
                        algo_values.extend(run_df.iloc[:, 0].tolist())
                        # update global min and mxa
                        env_global_min = min(env_global_min, min(algo_values))
                        env_global_max = max(env_global_max, max(algo_values))

                algo_data[algo] = algo_values

            # When there are multiple groups for comparison, their Wilconxon_p and A12 need to be stored
            Wilconxon_p = {}
            A12 = {}

            for i in range(len(algo_data)-1):
                # calculate Wilcoxon rank-sum and A12
                if len(algo_data[algorithms[0]]) > 0 and len(algo_data[algorithms[i+1]]) >0:
                    stat, p_value = stats.mannwhitneyu(algo_data[algorithms[0]], algo_data[algorithms[i+1]])
                    a12_value = calculate_a12(algo_data[algorithms[0]], algo_data[algorithms[i+1]])
                    Wilconxon_p[f'{algorithms[i+1]}'] = round(p_value, 3)
                    A12[f'{algorithms[i+1]}'] = round(a12_value, 3)




            '''
            if result['Wilcoxon_p'] < 0.05 and result['A12'] > 0.56:
                if result[f'{algorithms[0]}_median'] < result[f'{algorithms[1]}_median']:
                    if system !='h2':
                        result['label'] = 1
                    else:
                        result['label'] = -1
                else:
                    if system != 'h2':
                        result['label'] = -1
                    else:
                        result['label'] = 1
            else:
                result['label'] = 0
            '''


            for algo in algorithms:
                algo_values = algo_data[algo]
                mean = round(np.mean(algo_values), 3)
                std = round(np.std(algo_values), 3)
                iqr = round(calculate_iqr(algo_values), 3)



                # para1 25% as starting point
                norm_25 = round((np.percentile(algo_values, 25) - env_global_min) / (env_global_max - env_global_min) * 100, 4)

                # para2 IQR distance
                norm_75 = round(
                    (np.percentile(algo_values, 75) - env_global_min) / (env_global_max - env_global_min) * 100, 4)
                norm_IQR = norm_75 - norm_25

                # para3 starting point to ending point, 25%-75% add 50% point
                norm_50 = round((np.percentile(algo_values, 50) - env_global_min) / (env_global_max - env_global_min) * 100, 4)



                result = {
                    'Workload': env,
                    'Algorithm': algo,
                    'Mean (Std)': f'{mean} ({std})',
                }
                if algo == algorithms[0]:
                    result['$\hat{A}_{12}$ ($p$ value)'] = None
                else:
                    result['$\hat{A}_{12}$ ($p$ value)'] = f'{A12[algo]} ($p$ < 0.001)' if Wilconxon_p[algo] < 0.001 else f'{A12[algo]} ($p$ = {Wilconxon_p[algo]})'
                    #result['$\hat{A}_{12}$ ($p$ value)'] = f'{A12[algo]} ({Wilconxon_p[algo]})'

                #result[' '] = '\quart{' + str(norm_25) + '}{' + str(norm_IQR) + '}{' + str(norm_50) + '}{100}'

                results_list.append(result)

        # transform the list as DataFrame, and save to CSV
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f'RQ3/{system}.csv', index=False)


def main():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    #systems = ['lrzip', 'xz']
    # pair comparison, algorithm itself, variant-1, variant-2
    compared_algorithms = ['DLiSA-0.3', 'DLiSA-I', 'DLiSA-II']
    runs = 100
    collect_data(systems, compared_algorithms, runs)


if __name__ == "__main__":
    main()
