import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


plt.rcParams['text.usetex'] = False
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
# 在macOS中指定Times New Roman字体的典型路径
font_path = '/System/Library/Fonts/Supplemental/Times New Roman.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
def find_top_k_configs(configs, perfs, configs_ids, top_k=10, optimization_goal='minimum'):

    if optimization_goal == 'minimum':
        top_indices = perfs.iloc[:, 0].argsort()[:top_k]
    else:
        top_indices = perfs.iloc[:, 0].argsort()[::-1][:top_k]

    if configs.shape[1] == 1 and configs_ids.shape[1] == 1:
        top_k_configs = [np.array([val]) for val in configs.iloc[top_indices, 0]]  # 转换为包含单元素数组的列表
        top_k_configs_ids = configs_ids.iloc[top_indices, 0].tolist()
    else:
        top_k_configs = [np.array(row) for row in configs.iloc[top_indices].values]
        top_k_configs_ids = configs_ids.iloc[top_indices, 0].tolist()

    return top_k_configs, top_k_configs_ids


order_file = f'../order_files/order_kanzi_{82}.json'

if os.path.exists(order_file):
    with open(order_file, 'r') as f:
        data_files = json.load(f)

current_workload_file = data_files[7]
data = pd.read_csv(os.path.join('../dataset/kanzi', current_workload_file))
perf_space = data.iloc[:, -1].values

his_pop_configs = []
his_pop_perfs = []
his_pop_ids = []
optimized_config_And_id_folder = '../results/kanzi/tuning_results/DLiSA-0.3/optimized_pop_config_run_82'
optimized_perf_folder = '../results/kanzi/tuning_results/DLiSA-0.3/optimized_pop_perf_run_82'
for i, csv_file in enumerate(data_files[0:7]):
    workload_config_name = csv_file.replace('.csv', '_config.csv')
    workload_ids_name = csv_file.replace('.csv', '_indices.csv')
    workload_perfs_name = csv_file.replace('.csv', '_perf.csv')
    configs = pd.read_csv(os.path.join(optimized_config_And_id_folder, workload_config_name))
    his_pop_configs.append(configs)
    ids = pd.read_csv(os.path.join(optimized_config_And_id_folder, workload_ids_name))
    his_pop_ids.append(ids)
    perfs = pd.read_csv(os.path.join(optimized_perf_folder, workload_perfs_name))
    his_pop_perfs.append(perfs)

all_local_optima_configs = []
all_local_optima_ids = []

# collect those configs that perform better from each historical workload environment
for configs, perfs, configs_ids in zip(his_pop_configs, his_pop_perfs, his_pop_ids):

    # local_optima_indices = self.find_local_optima(config, perf, config_ids)
    top_k_configs, top_k_configs_ids = find_top_k_configs(configs, perfs, configs_ids, top_k=10)
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
    for i, env_configs in enumerate(his_pop_configs):
        if tuple(config) in [tuple(cfg) for cfg in env_configs]:
            latest_env = i
    latest_env_num[tuple(config)] = latest_env

# Calculate the compound weight of each considered configs
compound_weights = []
for config, count in zip(unique_optima_configs, counts):
    repeat_weight = count / len(his_pop_ids)  # 重复出现频率权重
    latest_weight = (1 + latest_env_num[tuple(config)]) / len(his_pop_ids)  # 最新环境权重
    compound_weight = repeat_weight + latest_weight
    compound_weights.append(compound_weight)

ids = list(config_to_id_mapping.values())

perf_values = [perf_space[id] for id in ids]

combined = list(zip(ids, compound_weights, perf_values))
combined_sorted = sorted(combined, key=lambda x: x[1])

sorted_ids, sorted_weights, sorted_perf_values = zip(*combined_sorted)

x_indices = range(1, len(sorted_ids) + 1)


fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制所有点的权重
ax1.set_xlabel('Sorted Configuration by Weight', fontsize = 24)
ax1.set_ylabel('Weight', color='#00247D', fontsize = 26)
ax1.set_ylim(0, 1)
print(sorted_weights)
ax1.plot(x_indices, sorted_weights, label='Weight', marker='o', linestyle='none', color='#00247D')
ax1.tick_params(axis='y', labelcolor='#00247D')


ax2 = ax1.twinx()
ax2.set_ylabel('Runtime', color='#C8102E', fontsize = 26)
ax2.set_ylim(0, 1)
print(sorted_perf_values)
ax2.plot(x_indices, sorted_perf_values, label='Runtime',  marker='^', linestyle='none', color='#C8102E')
ax2.tick_params(axis='y', labelcolor='#C8102E')



selected_ids = [1132, 162, 827, 1128, 1139, 3636, 1596, 1141, 3613, 1344, 1081, 227, 2144, 1288, 2624, 2401, 80, 1392, 1391, 78]

selected_x_indices = [sorted_ids.index(id) + 1 for id in selected_ids]  # +1 因为x_indices是从1开始的



for i, x_idx in enumerate(selected_x_indices):
    ax2.scatter(x_idx, sorted_perf_values[x_idx - 1], marker='^', edgecolor='green', linewidth=1, s=60)



for i, x_idx in enumerate(selected_x_indices):

    selected_weight = sorted_weights[x_idx - 1]
    selected_perf_value = sorted_perf_values[x_idx - 1]
    ax1.plot([x_idx, x_idx], [selected_weight, selected_perf_value], 'g--', linewidth=1, zorder=1)


fig.tight_layout()
plt.grid(True)
plt.show()
plt.savefig('Discussion/kanzi_weight.pdf')




