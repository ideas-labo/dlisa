import os
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

class GeneticAlgorithm:
    def __init__(self, pop_size, mutation_rate, crossover_rate, optimization_goal):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.evaluated_configs = []
        self.evaluated_configs_ids = []
        self.evaluated_configs_to_perfs = {}
        self.optimization_goal = optimization_goal

    def run(self, init_pop_config, init_pop_config_ids, config_space, perf_space, max_generation, environmental_selection_type, selected_algorithm, run_no, system, environment_name):

        parent_configs = init_pop_config.copy()
        parent_ids = init_pop_config_ids.copy()
        parent_perfs, parent_ids = self.evaluate(parent_ids, parent_configs, perf_space)

        output_folder_pop = 'results/' + system + '/tuning_results/' + selected_algorithm + '/optimized_pop_perf_run_' + str(
            run_no) + f'/evolutionary_process_in_{environment_name}'

        if not os.path.exists(output_folder_pop):
            os.makedirs(output_folder_pop)


        # Save the initial population
        np.savetxt(os.path.join(output_folder_pop, 'generation_0_config.csv'), parent_configs,
                   fmt='%f', delimiter=',')
        np.savetxt(os.path.join(output_folder_pop, f'generation_0_perf.csv'), parent_perfs,
                   delimiter=',')
        np.savetxt(os.path.join(output_folder_pop, 'generation_0__indices.csv'), parent_ids,
                   delimiter=',', fmt='%d')

        for i in range(max_generation):
            offspring_configs, offspring_ids = self.generate_offspring_by_cro_mut(parent_perfs, config_space, parent_configs)
            offspring_perfs, offspring_ids = self.evaluate(offspring_ids, offspring_configs, perf_space)

            combined_population = np.vstack((parent_configs, offspring_configs))
            combined_performance = np.concatenate((parent_perfs, offspring_perfs))
            combined_indices = np.concatenate((parent_ids, offspring_ids))

            if environmental_selection_type == 'LiDOS_selection':

                selected_indices = self.LiDOS_selection(combined_population, combined_performance, combined_indices)
            else:

                if self.optimization_goal == 'minimum':
                    selected_indices = np.argsort(combined_performance)[:self.pop_size]
                else:
                    selected_indices = np.argsort(combined_performance)[::-1][:self.pop_size]

            # Due to the environment selection mentioned above, the saved populations must be sorted in order of performance,
            # so the first row is selected as the best in the statistics.

            parent_configs = combined_population[selected_indices]
            parent_perfs = combined_performance[selected_indices]
            parent_ids = combined_indices[selected_indices]

            # Save the population in each generation
            np.savetxt(os.path.join(output_folder_pop, f'generation_{i+1}_config.csv'), parent_configs,
                       fmt='%f', delimiter=',')
            np.savetxt(os.path.join(output_folder_pop, f'generation_{i+1}_perf.csv'), parent_perfs,
                       delimiter=',')
            np.savetxt(os.path.join(output_folder_pop, f'generation_{i+1}__indices.csv'), parent_ids,
                       delimiter=',', fmt='%d')

        # return the config, perf, id in the final generation
        optimized_pop_configs = parent_configs
        optimized_pop_perfs = parent_perfs
        optimized_pop_indices = parent_ids
        return optimized_pop_configs, optimized_pop_perfs, optimized_pop_indices, self.evaluated_configs_to_perfs

    def evaluate(self, population_ids, population_configs, perf_space):
        performance = []

        if self.optimization_goal == 'minimum':
            lower_quartile_performance = np.percentile(perf_space, 75)
        else:
            lower_quartile_performance = np.percentile(perf_space, 25)

        for idx, individual_config in zip(population_ids, population_configs):
            if not (any(np.array_equal(individual_config, evaluated_config) for evaluated_config in self.evaluated_configs) and idx in self.evaluated_configs_ids):
                if idx != -1:
                    perf = perf_space[idx]
                else:
                    noise = np.random.uniform(0.001, 0.009)
                    perf = lower_quartile_performance + noise
                performance.append(perf)
                self.evaluated_configs.append(individual_config)
                self.evaluated_configs_ids.append(idx)
                # record those config and corresponding perf that are evaluated
                self.evaluated_configs_to_perfs[tuple(individual_config)] = perf

        return performance, population_ids

    def generate_offspring_by_cro_mut(self, parent_perfs, config_space, parent_configs):
        offspring_configs = []
        offspring_ids = []
        while len(offspring_configs) < self.pop_size:

            parent1_idx = self.tournament_selection(parent_perfs)
            parent2_idx = self.tournament_selection(parent_perfs)

            while parent1_idx == parent2_idx:
                parent2_idx = self.tournament_selection(parent_perfs)

            parent1 = parent_configs[parent1_idx]
            parent2 = parent_configs[parent2_idx]

            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1, config_space)
            child2 = self.mutate(child2, config_space)

            # distribute id for new generated offspring
            matches = np.where((config_space == child1).all(axis=1))[0]
            child1_idx = matches[0] if matches.size > 0 else -1
            matches = np.where((config_space == child2).all(axis=1))[0]
            child2_idx = matches[0] if matches.size > 0 else -1

            if self.is_valid_offspring(child1, parent_configs, offspring_configs):
                offspring_configs.append(child1)
                offspring_ids.append(child1_idx)
            if len(offspring_configs) < self.pop_size and self.is_valid_offspring(child2, parent_configs, offspring_configs):
                offspring_configs.append(child2)
                offspring_ids.append(child2_idx)

        return offspring_configs, offspring_ids

    def tournament_selection(self, performance):

        candidates = np.random.choice(len(performance), 2, replace=False)
        if self.optimization_goal == 'minimum':
            winner_idx = candidates[0] if performance[candidates[0]] < performance[candidates[1]] else candidates[1]
        else:
            winner_idx = candidates[0] if performance[candidates[0]] > performance[candidates[1]] else candidates[1]
        return winner_idx

    def crossover(self, parent1, parent2):
        # single point crossover
        if np.random.rand() < self.crossover_rate:
            cross_point = np.random.randint(1, len(parent1))
            child1 = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
            child2 = np.concatenate((parent2[:cross_point], parent1[cross_point:]))
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        return child1, child2

    def mutate(self, individual, config_space):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                unique_values = np.unique(config_space[:, i])
                individual[i] = np.random.choice(unique_values)
        return individual

    def is_valid_offspring(self, offspring_config, population_configs, offspring_configs):
        # validated examination
        is_not_in_population = not any(
            np.array_equal(offspring_config, parent_config) for parent_config in population_configs)
        is_not_evaluated = not any(
            np.array_equal(offspring_config, evaluated_config) for evaluated_config in self.evaluated_configs)
        is_not_in_offspring = not any(
            np.array_equal(offspring_config, other_offspring_config) for other_offspring_config in offspring_configs)

        return is_not_in_population and is_not_evaluated and is_not_in_offspring

    def find_nearest_neighbors(self, index, population, k):
        distances = cdist([population[index]], population)[0]
        distances[index] = np.inf
        neighbors_indices = np.argsort(distances)[:k]
        return neighbors_indices

    def generate_multi_objective_scores(self, combined_population, combined_performance, k=5):
        num_individuals = len(combined_population)
        combined_objectives = []

        for i in range(num_individuals):
            neighbors_indices = self.find_nearest_neighbors(i, combined_population, k)
            max_diff_perf = max(combined_performance[neighbors_indices], key=lambda x: abs(x - combined_performance[i]))

            g1 = combined_performance[i] + max_diff_perf
            g2 = combined_performance[i] - max_diff_perf
            combined_objectives.append((g1, g2))

        return combined_objectives

    def fast_non_dominated_sort(self, objectives):

        S = [[] for _ in range(len(objectives))]
        n = [0 for _ in range(len(objectives))]
        rank = [0 for _ in range(len(objectives))]
        front = [[]]
        for p in range(len(objectives)):
            for q in range(len(objectives)):
                if self.dominates(objectives[p], objectives[q]):
                    S[p].append(q)
                elif self.dominates(objectives[q], objectives[p]):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                front[0].append(p)
        i = 0
        while front[i]:
            next_front = []
            for p in front[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            front.append(next_front)
            i += 1
        return front[:-1]  # Last front will be empty

    def crowding_distance_assignment(self, objectives, front):
        num_objects = len(objectives[0])
        distances = [0] * len(front)

        if len(front) <= 1:
            return [float('inf')] * len(front)

        for m in range(num_objects):
            sorted_front = sorted(front, key=lambda i: objectives[i][m])
            distances[front.index(sorted_front[0])] = float('inf')
            distances[front.index(sorted_front[-1])] = float('inf')

            min_obj = objectives[sorted_front[0]][m]
            max_obj = objectives[sorted_front[-1]][m]

            if max_obj == min_obj:
                continue

            for i in range(1, len(sorted_front) - 1):
                distances[front.index(sorted_front[i])] += (objectives[sorted_front[i + 1]][m] -
                                                            objectives[sorted_front[i - 1]][m]) / (
                                                                   max_obj - min_obj)

        return distances

    def dominates(self, obj1, obj2):
        return all(o <= p for o, p in zip(obj1, obj2)) and any(o < p for o, p in zip(obj1, obj2))

    def LiDOS_selection(self, combined_population, combined_performance, combined_indices):

        combined_objectives = self.generate_multi_objective_scores(combined_population, combined_performance, k=5)

        if self.optimization_goal == 'maximum':
            combined_objectives = [(-g1, -g2) for g1, g2 in combined_objectives]

        fronts = self.fast_non_dominated_sort(combined_objectives)
        new_population = []
        new_population_objs = []
        new_indices = []

        for front in fronts:
            if len(new_population) + len(front) <= self.pop_size:
                new_population.extend([combined_population[i] for i in front])
                new_population_objs.extend([combined_performance[i] for i in front])
                new_indices.extend([i for i in front])
                if len(new_population) == self.pop_size:
                    break
            else:
                # Compute crowding distance for this front as it will exceed the population size
                crowding_distances = self.crowding_distance_assignment(combined_objectives, front)
                # Sort based on rank and crowding distance
                sorted_indices_by_crowding_distance = sorted(range(len(crowding_distances)),
                                                             key=lambda x: -crowding_distances[x])

                remaining_space = self.pop_size - len(new_population)

                selected_indices = [front[i] for i in sorted_indices_by_crowding_distance[:remaining_space]]

                new_population.extend([combined_population[i] for i in selected_indices])
                new_population_objs.extend([combined_performance[i] for i in selected_indices])
                new_indices.extend([i for i in selected_indices])
                break

        new_population_objs = np.array(new_population_objs)
        new_indices = np.array(new_indices)
        if self.optimization_goal == 'minimum':
            sorted_indices = np.argsort(new_population_objs)[:self.pop_size]
        else:
            sorted_indices = np.argsort(new_population_objs)[::-1][:self.pop_size]

        selected_indices = new_indices[sorted_indices]
        return selected_indices

