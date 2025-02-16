# Distilled Lifelong Self-Adaptation for Configurable Systems

> Modern configurable systems provide tremendous opportunities for engineering future intelligent software systems.
A key difficulty thereof is how to effectively self-adapt the configuration of a running system such that its performance (e.g., runtime and throughput) can be optimized under time-varying workloads. This unfortunately remains unaddressed in existing approaches as they either overlook the available past knowledge or rely on static exploitation of past knowledge without reasoning the usefulness of information when planning for self-adaptation. In this paper, we tackle this challenging problem by proposing **DLiSA**, a framework that self-adapts configurable systems. **DLiSA** comes with two properties: firstly, it supports lifelong planning, and thereby the planning process runs continuously throughout the lifetime of the system, allowing dynamic exploitation of the accumulated knowledge for rapid adaptation. Secondly, the planning for a newly emerged workload is boosted via distilled knowledge seeding, in which the knowledge is dynamically purified such that only useful past configurations are seeded when necessary, mitigating misleading information.

> 
> Extensive experiments suggest that the proposed **DLiSA** significantly outperforms state-of-the-art approaches, demonstrating a performance improvement of up to 229\% and a resource acceleration of up to 2.22x on generating promising adaptation configurations.

This repository contains the **key codes**, **full data used**,  and **raw experiment results** for the paper.

# Documents

## dataset
The `dataset` folder contains configuration datasets for the following **9 subject systems**, each stored in a separate subfolder. The name of each subfolder corresponds to the system it represents. 

| System  | Language  | Domain            | Performance | #O   | #C   | #W  | Optimization Goal |
|---------|-----------|-------------------|-------------|------|------|-----|-------------------|
| JUMP3R  | Java      | Audio Encoder     | Runtime     | 16   | 4196 |  6  | Minimization      |
| KANZI   | Java      | File Compressor   | Runtime     | 24   | 4112 |  9  | Minimization      |
| DCONVERT| Java      | Image Scaling     | Runtime     | 18   | 6764 | 12  | Minimization      |
| H2      | Java      | Database          | Throughput  | 16   | 1954 |  8  | Maximization      |
| BATLIK  | Java      | SVG Rasterizer    | Runtime     | 10   | 1919 | 11  | Minimization      |
| XZ      | C/C++     | File Compressor   | Runtime     | 33   | 1999 | 13  | Minimization      |
| LRZIP   | C/C++     | File Compressor   | Runtime     | 11   | 190  |  13 | Minimization      |
| X264    | C/C++     | Video Encoder     | Runtime     | 25   | 3113 |  9  | Minimization      |
| Z3      | C/C++     | SMT Solver        | Runtime     | 12   | 1011 | 12  | Minimization      |

- **#O**: Number of options.
- **#C**: Number of configurations.
- **#W**: Number of workloads tested.
- **Optimization Goal**: Whether the goal is **minimization** (e.g., reducing runtime) or **maximization** (e.g., increasing throughput).

Each system folder contains **6 ~ 13 workloads**, depending on the system. The data is stored in `.csv` format, where each CSV file follows the below structure:

- **Columns (1 to n-1):** Configuration options, which can be either discrete or continuous values.
- **Column n:** Performance objective, represented as a numeric value (e.g., runtime, throughput).


## order_files
The `order_files` folder contains the execution order of workloads for each system over **100 independent runs**. 


## results
The `results` folder contains the **optimization results** for each system. Each system has its own subfolder where the corresponding optimization outcomes are stored. Within each system's folder, the `tuning_results` directory stores the detailed optimization results for **all compared algorithms**, including both state-of-the-art baselines and our proposed method.

#### Algorithm Comparison

- **D-SOGA, FEMOSAA, LiDOS, Seed-EA**: State-of-the-art (SOTA) baseline algorithms for comparison.
- **DLiSA-0.x**: Our proposed **DLiSA** method, evaluated with different hyperparameter $\alpha$ settings.
- **DLiSA-I, DLiSA-II**: Two ablation study variants of **DLiSA**.

#### Folder Structure and Example
Each algorithm folder contains results from **100 independent runs**, where the optimization process is stored in separate directories:

    results/
    │── batlik/
    │   ├── tuning_results/
    │   │   ├── D-SOGA/
    │   │   │   ├── optimized_pop_config_run_0/
    │   │   │   │   ├── corona_config.csv
    │   │   │   │   └── corona_indices.csv
    │   │   │   ├── optimized_pop_perf_run_0/
    │   │   │   │   ├── corona_perf.csv
    │   │   │   │   ├── corona_time.csv
    │   │   │   │   └── evolutionary_process_in_corona/
    │   │   │   │   │   ├── generation_0_indices.csv 
    │   │   │   │   │   ├── generation_0_config.csv
    │   │   │   │   │   ├── generation_0_perf.csv
    │   │   │   │   │   ├── ...    
    


#### Directory Breakdown

- **`optimized_pop_config_run_{run}/`**: Stores the **optimized configurations** for the given run.
  - `corona_config.csv` stores the configuration details, while `corona_indices.csv` stores additional indices or data related to the configurations.

- **`optimized_pop_perf_run_{run}/`**: Stores the corresponding performance results for the optimized configurations in the corresponding run.
  - `corona_perf.csv` stores the performance of the optimized configuration, while `corona_time.csv` records execution time of the algorithm for processing corona workload.

- **`evolutionary_process_in_corona/`**: Stores the **intermediate data** for each generation during the optimization process. This folder contains the data of all generations throughout the evolutionary process for the **corona** workload. 
  - `generation_0_config.csv` stores the configurations for specified generation.
  - `generation_0_perf.csv` stores the corresponding performance values for the configurations.
  - `generation_0_indices.csv` stores additional indices or metadata associated with the configuration.

Since all algorithms are **population-based methods**, the results are stored in `.csv` format, representing the final optimized population.




## statistical tool
The `statistical tool` folder contains scripts for conducting statistical analyses related to **all research questions (RQs)** and their corresponding results.

## supplementary file
The `supplementary file` contains the specific workloads for 9 subjected systems and detailed experimental results for RQ2 and RQ3.

## python files
- **main.py**:
  the *main program* for using DLiSA, which automatically reads data from csv files, implements *Knowledge Distillation* and *Evolutionary Planning*, and saves the corresponding results and data.
- **Adaptation_Optimizer.py**
  the program for running *Knowledge Distillation*.
- **Genetic_Algorithm.py**
  the program for running *Evolutionary Planning*
- **requirements.txt**
  the required packages for running main.py.

# Prerequisites and Installation
1. Download all the files into the same folder/clone the repository.

2. Install the specified version of Python:
   - The codes have been tested and is **recommended to run with Python 3.9**, Other versions might cause errors.
   - Using other versions may require adjusting dependency versions; otherwise, package compatibility issues may occur.

3. Using the command line: cd to the folder with the codes, and install all the required packages by running:

        pip install -r requirements.txt
        
4. Install R: R is required to run the experiments. Please follow the installation instructions for your operating system at [R Project](https://cran.r-project.org/mirrors.html).

# Run *DLiSA*

- **Command line**: cd to the folder with the codes, input the command below, and the rest of the process will fully automated.

      python main.py
- **Python IDE (e.g., Pycharm)**: Open the *main.py* file on the IDE, and simply click 'Run'.

# Main Program Overview
The **main.py** script is structured with several key components:

- **Parameters Settings**: Configure essential parameters like the number of independent runs, maximum generations for GA, and population size.
- **Systems**: Define the list of systems to be optimized.
- **Compared Algorithms**: Specify which algorithms to compared for system performance tuning.
- **Optimization Goal**: Set the optimization objective for each (e.g., maximize throughput or minimize runtime).

# Example Setup for a Quick Demo
Modify these parameters in **main.py** to suit your optimization needs:

- **run**: Number of independent optimization runs (e.g., **run = 100**).
- **max_generation**: Maximum number of generations for the evolutionary process (e.g., **max_generation = 3**).
- **pop_size**: Population size for the Genetic Algorithm (e.g., **pop_size = 20**).
- **mutation_rate**: Probability of mutation in the evolutionary process (e.g., **mutation_rate = 0.1**).
- **crossover_rate**: Probability of crossover (e.g., **crossover_rate = 0.9**).
- **systems**: List of systems for optimization (e.g., **systems = ['batlik', 'kanzi']**).
- **compared_algorithm**: Algorithms for comparison (e.g.,**compared_algorithm = ['FEMOSAA', 'Seed-EA', 'D-SOGA', 'DLiSA-0.3']** )

# Evaluation

After setting the necessary parameters and successfully running the project, statistical analysis scripts in the `statistical tool` folder can be used to compare the performance of all algorithms.

## RQ1: How effective is DLiSA against state-of-the-art approaches?

The script **`statistical tool/RQ1_Scott-Knott.py`** performs **Scott-Knott** ranking analysis to evaluate the performance of different algorithms on the specified systems. The results are stored in the **`statistical tool/RQ1/`** directory, with each system's ranking saved in a CSV file: `statistical tool/RQ1/{system_name}.csv`

### Stored Format

Each CSV file follows the structure shown below:

| Workload | Algorithm  | r | Mean (Std)    |
|----------|------------|---|---------------|
| corona   | FEMOSAA    | 3 | 0.953 (0.043) |
| corona   | Seed-EA    | 2 | 0.912 (0.031) |
| corona   | D-SOGA     | 2 | 0.912 (0.024) |
| corona   | LiDOS      | 2 | 0.916 (0.025) |
| corona   | DLiSA-0.3  | 1 | 0.907 (0.014) |

- **Column `r`**: Represents the ranking of each algorithm, where **rank 1 indicates the best performance**.
- **Column `Mean (Std)`**: Displays the **mean performance value** across **100 independent runs**, along with the **standard deviation** in parentheses.

### Expected Outcome

The primary goal of our study is to **maximize the frequency at which DLiSA ranks 1**, demonstrating its superior performance compared to SOTA algorithms. In order to make a more intuitive comparison, **`RQ1_Scott-Knott.py`** also prints the number of times each algorithm achieves **rank 1** directly in the console. This allows for an easy assessment of which algorithm performs best across different workloads and systems. A simple snapshot of the execution is shown below:

![](https://raw.githubusercontent.com/ideas-labo/dlisa/refs/heads/main/RQ1-snapshot.jpg)

By implementing this script, you can **quantitatively compare** optimization methods and validate the effectiveness of **DLiSA** against SOTA algorithms.

## RQ2: How efficient is DLiSA compared with others?

The script **`statistical tool/RQ2_efficiency_analysis.py`** performs convergence analysis to evaluate the efficacy of different algorithms on the specified systems. Please refer to the paper for the detailed calculation process. The results are stored in the **`statistical tool/RQ2/`** folder.

### Stored Format  

The results are stored in CSV files, comparing the efficiency (convergence speed) of **DLiSA** against baseline algorithms on different systems. For example, the file: `convergence_kanzi_DLiSA-0.3_vs_FEMOSAA.csv` 
compares **DLiSA-0.3** (DLiSA with parameter $\alpha$=0.3) and **FEMOSAA** on the **Kanzi** system across different workloads.  

 
| workload       | baseline_b | baseline_T | compared_m | s           |
|----------------|------------|------------|------------|-------------|
| ambivert       | 80         | 1.8298     | 39         | 2.051282051 |
| artificl       | 80         | 0.1657     | 39         | 2.051282051 |
| deepfield      | 80         | 0.2926     | —          | -1          |
| enwik8         | 80         | 2.6246     | 38         | 2.105263158 |
| fannie_mae_500k| 80         | 1.8841     | 38         | 2.105263158 |
| large          | 80         | 0.6867     | 39         | 2.051282051 |
| misc           | 80         | 0.2426     | 39         | 2.051282051 |
| silesia        | 80         | 5.3858     | 38         | 2.105263158 |
| vmlinux        | 80         | 1.0921     | 39         | 2.051282051 |

#### **Column Explanation**
- **workload**: The tested workload in the Kanzi system.
- **baseline_b**: A baseline, b, is identified for the compared algorithm, representing the smallest number of measurements necessary for it to reach its best performance, i.e., **baseline_T**, averaging over 100 runs.
- **baseline_T**: The baseline algorithm’s best performance, averaging over 100 runs.
- **compared_m**: The smallest number of measurements required by the DLiSA to get a result that is equivalent or better than **baseline_T**.
- **s**: The speedup of **DLiSA** over its counterpart, where `s=-1` indicates that convergence was not achieved.  
  - In the paper, we use `s = N/A` to represent cases where DLiSA cannot achieve the **baseline_T** reached by its counterpart.


## RQ3: What benefits do ranked workload similarity analysis and weighted configuration seeding each provide?

The script **`statistical tool/RQ3_Non-parametric.py`** performs **non-parametric test** and **effect size test** for pairwise comparisons between DLiSA and its variants. The results are stored in the **`statistical tool/RQ3/`** folder.

### Stored Format

The results are stored in a CSV file with the following columns:

| Environment          | DLiSA-0.3_Mean | DLiSA-I_Mean | Wilcoxon_p | A12     | Label |
|----------------------|----------------|--------------|------------|---------|-------|
| beethoven.wav        | 2.5731         | 2.6442       | 0.02288209 | 0.59295 | 1     |
| dual-channel.wav     | 0.8462         | 0.9274       | 0.009418769| 0.6062  | 1     |
| helix.wav            | 1.3088         | 1.4313       | 0.006757337| 0.6108  | 1     |
| single-channel.wav   | 0.6425         | 0.6778       | 0.04459703 | 0.582   | 1     |
| speech.wav           | 1.0449         | 1.1274       | 0.000527182| 0.6416  | 1     |
| sweep.wav            | 0.2975         | 0.3074       | 0.002268491| 0.62165 | 1     |


- **Environment**: Represents different workloads tested.
- **{algorithm}mean**: Represents the mean value of 100 independent runs for each algorithm (e.g., **DLiSA-0.3_Mean**, **DLiSA-I_Mean**).
- **Wilcoxon_p**: p-value from the **Non-parametric test** indicating whether the difference between algorithms is statistically significant.
- **A12**: Represents the **effect size** indicating the strength of the algorithm's performance difference.
- **Label**: The label is **1** when the performance of DLiSA is better than its variants under the conditions:
  - **A12 >= 0.56**
  - **Wilcoxon_p <= 0.05**

### Expected Outcome

We expect that when the **A12 value is greater than or equal to 0.56** and the **Wilcoxon p-value is less than or equal to 0.05**, the performance of **DLiSA** is better than that of the comparison algorithms. In such cases, the **Label** will be marked as **1**. For detailed results, refer to [this link](https://github.com/ideas-labo/dlisa/blob/main/supplementary%20file/RQ3_summary/jump3r.pdf).

## RQ4: How does $\alpha$ affect DLiSA’s performance?

The script **`statistical tool/RQ4_Scott-Knott.py`** performs **Scott-Knott** ranking analysis to evaluate the performance of DLiSA with different parameter $\alpha$. The results are stored in the **`statistical tool/RQ4/`** directory, with each system's ranking saved in a CSV file: `statistical tool/R4/{system_name}.csv`.

## **Note on Reproducibility**  

Since the workload order is randomly shuffled across **100 independent runs**, and the optimization algorithms used are population-based methods, the **initial population** may vary in each execution. As a result, the reproduced results may exhibit slight statistical fluctuations compared to the reported data in the paper. However, these variations should not affect the overall conclusions of the study.

### **Execution Time Estimation**  
- Running `main.py` typically takes **around 3 hours** to complete.  
- The evaluation scripts for **RQ1 ~ RQ4** in the `statistical tool` folder generally take only **a few minutes** each.  
