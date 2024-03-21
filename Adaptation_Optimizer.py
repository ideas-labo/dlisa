import os
import random
from itertools import combinations
import numpy as np
import pandas as pd
import time


from Genetic_Algorithm import GeneticAlgorithm

class AdaptationOptimizer:
    def __init__(self, max_generation, pop_size, mutation_rate, crossover_rate, compared_algorithms, system, optimization_goal):
        self.max_generation = max_generation
        self.pop_size = pop_size
        self.compared_algorithms = compared_algorithms
        self.system = system
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.optimization_goal = optimization_goal


    def dynamic_optimization(self, data_folder, data_files, run_no):

        # Store the initial seed to allow the comparison algorithm to use the same initial seeds
        initial_seeds = None
        initial_seeds_ids = None

        for selected_algorithm in self.compared_algorithms:

            self.his_pop_configs = []  # Storing optimized populations of each workload environments (config space)
            self.his_pop_perfs = []  # Storing optimized populations of each workload environments (perf space)
            self.his_pop_ids = []  # Storing optimized populations of each workload environments (corresponding ids)
            self.his_envs_name = []  # Storing workload environment name
            self.his_evaluated_configs_to_perfs = []  # Dic: Storing evaluated config -> perf of each workload environment
            self.similarity_score = {}  # Storing the similarity_score

            # Only LiDOS has unique environmental selection strategy
            if selected_algorithm == 'LiDOS':
                environmental_selection_type = 'LiDOS_selection'
            else:
                environmental_selection_type = 'Traditional_selection'

            output_folder_pop_perf = 'results/' + self.system + '/tuning_results/' + selected_algorithm + '/optimized_pop_perf_run_' + str(run_no)
            output_folder_pop_config = 'results/' + self.system + '/tuning_results/' + selected_algorithm + '/optimized_pop_config_run_' + str(run_no)
            if not os.path.exists(output_folder_pop_perf):
                os.makedirs(output_folder_pop_perf)
            if not os.path.exists(output_folder_pop_config):
                os.makedirs(output_folder_pop_config)

            # Iterate each workload and tuning system under corresponding workload environment
            for i, csv_file in enumerate(data_files):
                environment_name = os.path.splitext(csv_file)[0]
                data = pd.read_csv(os.path.join(data_folder, csv_file))
                config_space = data.iloc[:, :-1].values
                perf_space = data.iloc[:, -1].values

                if i < 2:
                    self.similarity_score[environment_name] = 0

                # Record the running time of different algorithm
                time_start = time.time()

                if i == 0:
                    # For the first workload environment, the random population is initialized and saved
                    if initial_seeds is None:
                        initial_seeds, initial_seeds_ids = self.initialize_population(config_space, self.pop_size)
                    init_pop_config = initial_seeds
                    init_pop_config_ids = initial_seeds_ids
                else:
                    # For environments that are not first workload environments, different policies will make different response strategies
                    # to generate populations for the new workload environment
                    init_pop_config, init_pop_config_ids = self.generate_next_population(config_space, selected_algorithm, environment_name)


                # Evolutionary algorithm is used to obtain superior config
                ga = GeneticAlgorithm(self.pop_size, self.mutation_rate, self.crossover_rate, self.optimization_goal)
                optimized_pop_configs, optimized_pop_perfs, optimized_pop_indices, evaluated_configs_to_perfs = ga.run(init_pop_config, init_pop_config_ids, config_space, perf_space, self.max_generation, environmental_selection_type, selected_algorithm, run_no, self.system, environment_name)

                time_end = time.time()
                time_sum = np.array([time_end - time_start])

                # Save the optimized results for experimental data analysis
                np.savetxt(os.path.join(output_folder_pop_config, f'{environment_name}_config.csv'), optimized_pop_configs, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(output_folder_pop_perf, f'{environment_name}_perf.csv'), optimized_pop_perfs, delimiter=',')
                np.savetxt(os.path.join(output_folder_pop_config, f'{environment_name}_indices.csv'), optimized_pop_indices, delimiter=',', fmt='%d')
                np.savetxt(os.path.join(output_folder_pop_perf, f'{environment_name}_time.csv'), time_sum, delimiter=',')

                # update or save related data, which will be reused in the subsequent optimization
                self.his_evaluated_configs_to_perfs.append(evaluated_configs_to_perfs)
                self.his_pop_configs.append(optimized_pop_configs)
                self.his_pop_perfs.append(optimized_pop_perfs)
                self.his_pop_ids.append(optimized_pop_indices)
                self.his_envs_name.append(environment_name)

            df = pd.DataFrame([self.similarity_score])
            df.to_csv(os.path.join(output_folder_pop_perf, 'similarity_score.csv'), index_label=False)



    def initialize_population(self, config_space, required_size, existing_configs=None, existing_ids=None):

        ''' As the existing dataset did not test all the config to obtain corresponding perf, simply generate configs may not exist in the dataset
        -50% from dataset, 50% from randomly generate to guarantee algorithm running
        -remark: it's necessary to keep each config in population is unique
        -all the compared algorithm should keep consistent to guarantee fair comparison
        '''

        existing_configs_hashes = set(map(lambda x: hash(x.tobytes()), existing_configs)) if existing_configs is not None else set()
        existing_ids = set(existing_ids) if existing_ids is not None else set()

        pop_size_from_data = required_size // 2
        pop_configs_from_data = []
        pop_ids_from_data = []

        while len(pop_configs_from_data) < pop_size_from_data:
            idx = np.random.choice(len(config_space))
            config = config_space[idx]
            config_hash = hash(config.tobytes())
            if idx not in existing_ids:
                pop_configs_from_data.append(config)
                pop_ids_from_data.append(idx)

                existing_configs_hashes.add(config_hash)
                existing_ids.add(idx)
        pop_configs_from_data = np.array(pop_configs_from_data)

        pop_configs_from_random = []
        while len(pop_configs_from_random) < required_size - pop_size_from_data:
            config = np.array([np.random.choice(np.unique(config_space[:, i])) for i in range(config_space.shape[1])])
            config_hash = hash(config.tobytes())
            if config_hash not in existing_configs_hashes:
                pop_configs_from_random.append(config)
                existing_configs_hashes.add(config_hash)
        population_configs_from_random = np.array(pop_configs_from_random)

        # Distribute id for randomly generated configs (do not exist in dataset -> -1)
        pop_ids_from_random = []
        for config in population_configs_from_random:
            matches = np.where((config_space == config).all(axis=1))[0]
            id = matches[0] if matches.size > 0 else -1
            pop_ids_from_random.append(id)

        population_configs = np.vstack((pop_configs_from_data, population_configs_from_random))
        population_ids = np.concatenate((pop_ids_from_data, pop_ids_from_random))
        return population_configs, population_ids

    def generate_next_population(self, config_space, selected_algorithm, environment_name, beta=0.3):

        #  ************Stationary planner************
        if selected_algorithm == 'FEMOSAA':
            init_pop_config, init_pop_config_ids = self.initialize_population(config_space, self.pop_size)

        #  ************Seed-EA (Dynamic adaptation)************
        elif selected_algorithm == 'Seed-EA':
            init_pop_config_ids = self.his_pop_ids[-1]
            init_pop_config = self.his_pop_configs[-1]

        #  *******D-SOGA (Mixed Adaptation)**********
        elif selected_algorithm == 'D-SOGA':
            # 80% preserve and 20% randomly induce
            full_pop_size_his_configs = self.his_pop_configs[-1]
            full_pop_size_his_config_ids = self.his_pop_ids[-1]
            full_pop_size_his_config_ids = np.array(full_pop_size_his_config_ids).flatten()

            selected_indices = np.random.choice(self.pop_size, size=self.pop_size // 10 * 8, replace=False)
            memory_pop_config = full_pop_size_his_configs[selected_indices]
            memory_pop_config_ids = full_pop_size_his_config_ids[selected_indices]
            random_pop_config, random_pop_config_ids = self.initialize_population(config_space, self.pop_size // 10 * 2,
                                                                                  memory_pop_config,
                                                                                  memory_pop_config_ids)

            init_pop_config = np.vstack((memory_pop_config, random_pop_config))
            init_pop_config_ids = np.concatenate((memory_pop_config_ids, random_pop_config_ids))

        #  ************LiDOS (Dynamic adaptation)***************
        elif selected_algorithm == 'LiDOS':
            init_pop_config_ids = self.his_pop_ids[-1]
            init_pop_config = self.his_pop_configs[-1]

        #  ************The proposed algorithm************
        elif selected_algorithm =='DLiSA':
            if not self.his_pop_ids:
                raise ValueError("The list of his_pop_ids is empty")
            if len(self.his_pop_ids) == 1:
                init_pop_config, init_pop_config_ids = self.generate_next_population_based_medium_similarity(config_space)
            else:
                average_similarity = self.calculate_average_similarity(self.his_evaluated_configs_to_perfs, beta)
                self.similarity_score[environment_name] = average_similarity

                if average_similarity >= beta:
                    init_pop_config, init_pop_config_ids = self.generate_next_population_based_high_similarity(config_space)
                else:
                    init_pop_config, init_pop_config_ids = self.initialize_population(config_space, self.pop_size)

        #**********sensitivity to threshold*******************
        elif selected_algorithm =='DLiSA-0.7':
            init_pop_config, init_pop_config_ids = self.generate_next_population(config_space, 'DLiSA', environment_name,0.70)
        elif selected_algorithm =='DLiSA-0.8':
            init_pop_config, init_pop_config_ids = self.generate_next_population(config_space, 'DLiSA', environment_name,0.80)
        elif selected_algorithm =='DLiSA-0.9':
            init_pop_config, init_pop_config_ids = self.generate_next_population(config_space, 'DLiSA', environment_name, 0.90)
        elif selected_algorithm =='DLiSA-0.1':
            init_pop_config, init_pop_config_ids = self.generate_next_population(config_space, 'DLiSA', environment_name, 0.10)
        elif selected_algorithm == 'DLiSA-0.2':
            init_pop_config, init_pop_config_ids = self.generate_next_population(config_space, 'DLiSA', environment_name,0.20)
        elif selected_algorithm =='DLiSA-0.3':
            init_pop_config, init_pop_config_ids = self.generate_next_population(config_space, 'DLiSA', environment_name, 0.30)
        elif selected_algorithm == 'DLiSA-0.4':
            init_pop_config, init_pop_config_ids = self.generate_next_population(config_space, 'DLiSA', environment_name,0.40)
        elif selected_algorithm == 'DLiSA-0.5':
            init_pop_config, init_pop_config_ids = self.generate_next_population(config_space, 'DLiSA', environment_name,0.50)
        elif selected_algorithm == 'DLiSA-0.6':
            init_pop_config, init_pop_config_ids = self.generate_next_population(config_space, 'DLiSA', environment_name,0.60)
        elif selected_algorithm == 'DLiSA-0.0':
            init_pop_config, init_pop_config_ids = self.generate_next_population(config_space, 'DLiSA', environment_name,0.00)


        # Ablation Experiment DLiSA-I (the first variant of DLiSA)
        elif selected_algorithm == 'DLiSA-I':
            if not self.his_pop_ids:
                raise ValueError("The list of his_pop_ids is empty")
            if len(self.his_pop_ids) == 1:
                init_pop_config, init_pop_config_ids = self.generate_next_population_based_medium_similarity(
                    config_space)
            else:
                average_similarity = self.calculate_average_similarity(self.his_evaluated_configs_to_perfs, beta)
                if average_similarity >= beta:
                    all_historical_configs = []
                    all_historical_ids = []

                    # collect those configs that perform better from each historical workload environment
                    for configs, perfs, configs_ids in zip(self.his_pop_configs, self.his_pop_perfs, self.his_pop_ids):
                        all_historical_configs.extend(configs)
                        all_historical_ids.extend(configs_ids)

                    # Count the number of occurrences of each relatively optimal configuration and convert to hashable form
                    hashable_optima_configs = [tuple(config) for config in all_historical_configs]
                    unique_optima_configs, counts = np.unique(hashable_optima_configs, axis=0, return_counts=True)

                    # hashable config->id
                    config_to_id_mapping = {tuple(config): id for config, id in
                                            zip(all_historical_configs, all_historical_ids)}

                    # Selection configs to transfer according to the probability
                    selected_config_tuples = []
                    selected_ids = []
                    if len(unique_optima_configs) >= self.pop_size:
                        # unique_optima_configs > pop_size, selecte configs according to probability
                        selected_indices = np.random.choice(len(unique_optima_configs), size=self.pop_size, replace=False)
                        for idx in selected_indices:
                            config_tuple = unique_optima_configs[idx]
                            selected_config_tuples.append(config_tuple)
                            # 转换为元组后用作字典的键
                            selected_ids.append(config_to_id_mapping[tuple(config_tuple)])
                    else:
                        # unique_optima_configs < pop_size, all the unique_optima_configs are selected
                        for config in unique_optima_configs:
                            selected_config_tuples.append(config)
                            selected_ids.append(config_to_id_mapping[tuple(config)])
                        # Compensate by generating some random configs
                        while len(selected_config_tuples) < self.pop_size:
                            idx = np.random.choice(len(config_space))
                            config = config_space[idx]
                            if idx not in selected_ids:
                                selected_config_tuples.append(config)
                                selected_ids.append(idx)

                    # adjust the format
                    init_pop_config = np.array([list(config) for config in selected_config_tuples])
                    init_pop_config_ids = selected_ids

                else:
                    init_pop_config, init_pop_config_ids = self.initialize_population(config_space, self.pop_size)


        # Ablation Experiment DLiSA-II (the second varitant of DLiSA)
        elif selected_algorithm == 'DLiSA-II':
            if not self.his_pop_ids:
                raise ValueError("The list of his_pop_ids is empty")
            if len(self.his_pop_ids) == 1:
                init_pop_config, init_pop_config_ids = self.generate_next_population_based_medium_similarity(
                    config_space)
            else:
                average_similarity = random.random()
                if average_similarity >= beta:
                    init_pop_config, init_pop_config_ids = self.generate_next_population_based_high_similarity(config_space)
                else:
                    init_pop_config, init_pop_config_ids = self.initialize_population(config_space, self.pop_size)

        return init_pop_config, init_pop_config_ids

    # Local stage weighting, focues on selecting the best config locally, N/2 configs
    def find_top_k_configs(self, configs, perfs, configs_ids, top_k=10):
        # find top k configs that perform better
        if self.optimization_goal == 'minimum':
            top_indices = np.argsort(perfs)[:top_k]
        else:
            top_indices = np.argsort(perfs)[::-1][:top_k]

        top_k_configs = [configs[i] for i in top_indices]
        top_k_configs_ids = [configs_ids[i] for i in top_indices]
        return top_k_configs, top_k_configs_ids

    def generate_next_population_based_high_similarity(self, config_space):

        all_local_optima_configs = []
        all_local_optima_ids = []

        # collect those configs that perform better from each historical workload environment
        for configs, perfs, configs_ids in zip(self.his_pop_configs, self.his_pop_perfs, self.his_pop_ids):
            # local_optima_indices = self.find_local_optima(config, perf, config_ids)
            top_k_configs, top_k_configs_ids = self.find_top_k_configs(configs, perfs, configs_ids, top_k=10)
            all_local_optima_configs.extend(top_k_configs)
            all_local_optima_ids.extend(top_k_configs_ids)

        # Count the number of occurrences of each relatively optimal configuration and convert to hashable form
        hashable_optima_configs = [tuple(config) for config in all_local_optima_configs]
        unique_optima_configs, counts = np.unique(hashable_optima_configs, axis=0, return_counts=True)

        # hashable config->id
        config_to_id_mapping = {tuple(config): id for config, id in
                                zip(all_local_optima_configs, all_local_optima_ids)}

        # Calculate the latest environment number in which each solution occurs
        latest_env_num = {}
        for config in unique_optima_configs:
            latest_env = -1
            for i, env_configs in enumerate(self.his_pop_configs):
                if tuple(config) in [tuple(cfg) for cfg in env_configs]:
                    latest_env = i
            latest_env_num[tuple(config)] = latest_env

        # Calculate the compound weight of each considered configs
        compound_weights = []
        for config, count in zip(unique_optima_configs, counts):
            repeat_weight = count / len(self.his_pop_ids)  # robustness weight
            latest_weight = (1 + latest_env_num[tuple(config)]) / len(self.his_pop_ids)  # timeliness weight
            compound_weight = repeat_weight + latest_weight
            compound_weights.append(compound_weight)

        # Calculate the probability of selection
        compound_weights = np.array(compound_weights)
        probabilities = compound_weights / compound_weights.sum()

        # Selection configs to transfer according to the probability
        selected_config_tuples = []
        selected_ids = []
        if len(unique_optima_configs) >= self.pop_size:
            # unique_optima_configs > pop_size, selecte configs according to probability
            selected_indices = np.random.choice(len(unique_optima_configs), size=self.pop_size, p=probabilities,
                                                replace=False)
            for idx in selected_indices:
                config_tuple = unique_optima_configs[idx]
                selected_config_tuples.append(config_tuple)

                selected_ids.append(config_to_id_mapping[tuple(config_tuple)])
        else:
            # unique_optima_configs < pop_size, all the unique_optima_configs are selected
            for config in unique_optima_configs:
                selected_config_tuples.append(config)

                selected_ids.append(config_to_id_mapping[tuple(config)])
            # Compensate by generating some random configs
            while len(selected_config_tuples) < self.pop_size:
                idx = np.random.choice(len(config_space))
                config = config_space[idx]
                if idx not in selected_ids:
                    selected_config_tuples.append(config)
                    selected_ids.append(idx)

        # adjust the format
        init_pop_config = np.array([list(config) for config in selected_config_tuples])
        init_pop_config_ids = selected_ids

        return init_pop_config, init_pop_config_ids

    def generate_next_population_based_medium_similarity(self, config_space):

        # top 50% configs from last workload environment are preserved
        full_pop_size_his_configs = self.his_pop_configs[-1]
        full_pop_size_his_config_ids = self.his_pop_ids[-1]

        full_pop_size_his_config_ids = np.array(full_pop_size_his_config_ids).flatten()
        similar_pop_config = full_pop_size_his_configs[:self.pop_size//2]
        similar_pop_config_ids = full_pop_size_his_config_ids[:self.pop_size//2]

        # the rest of 50% are compensate by randomly initialization
        random_pop_config, random_pop_config_ids = self.initialize_population(config_space, self.pop_size // 2, similar_pop_config, similar_pop_config_ids)

        init_pop_config = np.vstack((similar_pop_config, random_pop_config))
        init_pop_config_ids = np.concatenate((similar_pop_config_ids, random_pop_config_ids))

        return init_pop_config, init_pop_config_ids

    def calculate_similarity(self, env1, env2, common_solutions, beta):
        '''
        :param env1: Dic config->perf of env1
        :param env2: Dic config->perf of env2
        :param common_solutions: those solutions that are evaluated both in env1 and env2
        :return:
        '''

        if len(common_solutions) > self.pop_size * 0.25:
            total_pairs = 0
            consistent_pairs = 0
            for sol1, sol2 in combinations(common_solutions, 2):
                total_pairs += 1
                perf_env1_sol1 = env1[sol1]
                perf_env1_sol2 = env1[sol2]
                perf_env2_sol1 = env2[sol1]
                perf_env2_sol2 = env2[sol2]
                # match the ranking consistency
                if (perf_env1_sol1 > perf_env1_sol2) == (perf_env2_sol1 > perf_env2_sol2):
                    consistent_pairs += 1
            similarity_score = consistent_pairs / total_pairs if total_pairs > 0 else 0
        else:
            if beta == 0.0:
                beta = 0.3
            # without common solution (or just a very small percentage, <25%), random a relative small similarity score
            # conservative strategy, small similarity score may give a more likelihood to trigger random initialization
            similarity_score = random.uniform(0, beta)

        return similarity_score

    def calculate_average_similarity(self, his_evaluated_configs_to_perfs, beta):
        n = len(his_evaluated_configs_to_perfs)
        total_similarity = 0
        count = 0
        for i in range(n-1):
            env1_evaluated_configs_to_perfs = his_evaluated_configs_to_perfs[i]
            env2_evaluated_configs_to_perfs = his_evaluated_configs_to_perfs[i + 1]
            common_solutions = set(env1_evaluated_configs_to_perfs.keys()) & set(env2_evaluated_configs_to_perfs.keys())
            similarity = self.calculate_similarity(env1_evaluated_configs_to_perfs, env2_evaluated_configs_to_perfs, common_solutions, beta)
            total_similarity += similarity
            count += 1

        average_similarity = round(total_similarity / count, 2) if count > 0 else 0

        return average_similarity
