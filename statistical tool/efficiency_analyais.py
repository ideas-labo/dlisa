import os
import numpy as np
import pandas as pd

def find_first_index_of_optimal_performance(performance_data, optimal_value, system):

    if system == 'h2':
        indices = np.where(performance_data >= optimal_value)[0]
    else:
        indices = np.where(performance_data <= optimal_value)[0]

    return indices[0] + 1 if indices.size > 0 else None

def efficiency_analysis(systems, compared_algorithm, runs, baseline_algorithm, max_generation, base_path='../results'):

    system_results = {}
    for system in systems:

        results = []
        system_path = os.path.join(base_path, system, 'tuning_results')
        env_path = os.path.join(system_path, compared_algorithm, 'optimized_pop_perf_run_0')
        environments = sorted([f.replace('_perf.csv', '') for f in os.listdir(env_path) if f.endswith('_perf.csv')])

        for env in environments:

            all_perf_data_baseline = []
            all_perf_data_compared_algorithm = []

            for run in range(runs):
                perf_data_baseline = []
                perf_data_compared_algorithm = []

                for generation in range(max_generation + 1):
                    evolution_file_path_baseline = os.path.join(system_path, baseline_algorithm, f'optimized_pop_perf_run_{run}', f'evolutionary_process_in_{env}', f'generation_{generation}_perf.csv')
                    evolution_file_path_compared_algorithm = os.path.join(system_path, compared_algorithm, f'optimized_pop_perf_run_{run}', f'evolutionary_process_in_{env}', f'generation_{generation}_perf.csv')
                    if os.path.exists(evolution_file_path_baseline):
                        gen_data = pd.read_csv(evolution_file_path_baseline, header=None)

                        perf_data_baseline.extend(gen_data[0].iloc[::-1].tolist())
                    if os.path.exists(evolution_file_path_compared_algorithm):
                        gen_data = pd.read_csv(evolution_file_path_compared_algorithm, header=None)

                        perf_data_compared_algorithm.extend(gen_data[0].iloc[::-1].tolist())

                all_perf_data_baseline.append(perf_data_baseline)
                all_perf_data_compared_algorithm.append(perf_data_compared_algorithm)


            # Calculate average performance

            baseline_mean = np.mean(all_perf_data_baseline, axis=0)
            baseline_std = np.std(all_perf_data_baseline, axis=0)
            compared_mean = np.mean(all_perf_data_compared_algorithm, axis=0)
            compared_std = np.std(all_perf_data_compared_algorithm, axis=0)

            # Ensure convergence for plotting
            conv_avg_perf_data_baseline, conv_std_perf_data_baseline = ensure_convergence(baseline_mean, baseline_std, system)
            conv_avg_perf_data_compared, conv_std_perf_data_compared = ensure_convergence(compared_mean, compared_std, system)

            # Find the optimal performance value T and the number of evaluations b of the baseline algorithm
            optimal_value_baseline = conv_avg_perf_data_baseline[-1]
            b = find_first_index_of_optimal_performance(conv_avg_perf_data_baseline, optimal_value_baseline, system)

            # Find the number of evaluations m for which the comparison algorithms achieve the same performance T
            m = find_first_index_of_optimal_performance(conv_avg_perf_data_compared, optimal_value_baseline, system)

            # If the comparison algorithm does not reach the optimal performance of the baseline algorithm, s = -1
            if m is None:
                s = -1
            else:
                s = b / m if m else 0

            if s > 1:
                label = 1
            elif s == 1:
                label = 0
            elif 0 < s < 1:
                label = 999
            elif s == -1:
                label = -1

            results.append([env, b, optimal_value_baseline, m, s, label])

        system_results[system] = results


    for system, results in system_results.items():
        results_df = pd.DataFrame(results,
                                  columns=['workload', 'baseline_b', 'baseline_T', 'compared_m', 's', 'label'])
        results_df.to_csv(os.path.join(f'RQ2/convergence_{system}_{compared_algorithm}_vs_{baseline_algorithm}.csv'), index=False)




def ensure_convergence(mean_data, std_data, system):
    for i in range(1, len(mean_data)):
        if system == 'h2':
            if mean_data[i] < mean_data[i - 1]:
                mean_data[i] = mean_data[i - 1]
                std_data[i] = std_data[i - 1]
        else:
            if mean_data[i] > mean_data[i - 1]:
                mean_data[i] = mean_data[i - 1]
                std_data[i] = std_data[i - 1]
    return mean_data, std_data



def main():

    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']

    baseline_algorithm = 'D-SOGA'
    compared_algorithm = 'DLiSA-0.3' #'FEMOSAA', 'Seed-EA', 'D-SOGA', 'LiDOS',
    runs = 100
    max_generation = 3
    efficiency_analysis(systems, compared_algorithm, runs, baseline_algorithm, max_generation)




if __name__ == '__main__':
    main()


