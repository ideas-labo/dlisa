# Distilled Lifelong Self-Adaptation for Configurable Systems

> Modern configurable systems provide tremendous opportunities for engineering future intelligent software systems. A key difficulty thereof is how to effectively adapt the configuration of a running system such that its performance (e.g., runtime and throughput) can be optimized under time-varying workloads. This unfortunately remains unaddressed in existing approaches as they either waste the available past knowledge or fully trust the knowledge without mitigating the potentially misleading information when planning for self-adaptation. In this paper, we tackle this challenging problem by proposing **DLiSA**, a framework that self-adapts configurable systems. **DLiSA** comes with two unique properties: it supports lifelong planning and thereby the planning process runs continuously throughout the lifetime of the system, allowing exploitation of the accumulated knowledge for rapid adaptation. More importantly, the planning process for a newly emerged workload is boosted via distilled knowledge seeding: the knowledge is purified such that the most useful past configurations are seeded only when necessary, mitigating the impact of misleading information.
> 
> Extensive experiments suggest that the proposed **DLiSA** significantly outperforms state-of-the-art approaches, demonstrating a performance improvement of up to 255\% and a resource acceleration of up to 2.22x on generating promising adaptation configurations.

This repository contains the **key codes**, **full data used**,  and **raw experiment results** for the paper.

# Documents

- **dataset**:
  configuration datasets of 9 subject systems as specified in the paper.
- **order_files**:
  Sequence of workloads in 100 independent runs for 9 systems.
- **results**:
  contains the raw experiment results for all the research questions.
- **statistical tool**:
  contains statistical indicators for all research questions and corresponding statistical results.
- **supplementary file**:
  contains the specific workloads for 9 subjected systems and detailed experimental results for RQ2 and RQ3.
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
   the codes have been tested with **Python 3.7 - 3.9**, Other versions might cause errors.

3. Using the command line: cd to the folder with the codes, and install all the required packages by running:

        pip install -r requirements.txt

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

