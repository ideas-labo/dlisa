import os
import pandas as pd
from scipy import stats
import numpy as np

def calculate_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    return q3 - q1

def calculate_a12(group1, group2):
    n1, n2 = len(group1), len(group2)
    ranked = stats.rankdata(np.concatenate((group1, group2)))
    rank1 = ranked[:n1]
    a12 = ((rank1.sum() - (n1 * (n1 + 1) / 2)) / (n1 * n2))
    return a12 if a12 > 0.5 else 1 - a12

# 收集数据并进行统计
def collect_data(systems, algorithms, runs, base_path='../results'):


    for system in systems:
        # 存储结果的列表
        results_list = []
        system_path = os.path.join(base_path, system, 'tuning_results')
        env_path = os.path.join(system_path, algorithms[0], 'optimized_pop_perf_run_0')
        environments = sorted([f.replace('_perf.csv', '') for f in os.listdir(env_path) if f.endswith('_perf.csv')])

        for env in environments:
            algo_data = {}

            for algo in algorithms:
                algo_values = []
                for run in range(runs):
                    file_path = os.path.join(system_path, algo, f'optimized_pop_perf_run_{run}', f'{env}_perf.csv')
                    if os.path.exists(file_path):
                        run_df = pd.read_csv(file_path, header=None, nrows=1)
                        algo_values.extend(run_df.iloc[0].values)

                algo_data[algo] = algo_values

            if all(len(algo_data[algo]) > 0 for algo in algorithms):
                stat, p_value = stats.mannwhitneyu(algo_data[algorithms[0]], algo_data[algorithms[1]])
                a12_value = calculate_a12(algo_data[algorithms[0]], algo_data[algorithms[1]])

                if p_value < 0.05 and a12_value > 0.56:
                    if np.mean(algo_data[algorithms[0]]) < np.mean(algo_data[algorithms[1]]):
                        if system !='h2':
                            label = 1
                        else:
                            label = -1
                    else:
                        if system != 'h2':
                            label = -1
                        else:
                            label = 1
                else:
                    label = 0

                results = {
                    'Environment': env,
                    f'{algorithms[0]}_Mean': np.mean(algo_data[algorithms[0]]),
                    f'{algorithms[1]}_Mean': np.mean(algo_data[algorithms[1]]),
                    'Wilcoxon_p': p_value,
                    'A12': a12_value,
                    'Label': label
                }
                results_list.append(results)

        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f'RQ3/{system}_{algorithms[0]}_vs_{algorithms[1]}.csv', index=False)

def main():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    compared_algorithms = ['DLiSA-0.3', 'DLiSA-I']
    runs = 100
    collect_data(systems, compared_algorithms, runs)

if __name__ == "__main__":
    main()
