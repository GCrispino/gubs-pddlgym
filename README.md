# gubs-pddlgym

Implementation of algorithms eGUBS-VI and eGUBS-AO* for solving the GUBS criterion in PDDLGym environments.

This is the code used for the first set of experiments of the paper "GUBS criterion: arbitrary trade-offs between cost and probability-to-goal in stochastic planning based on Expected Utility Theory" ([link](https://www.sciencedirect.com/science/article/pii/S0004370222001886)), that compares the performance of algorithms eGUBS-VI and eGUBS-AO*.

Note: for the code used in the second set of experiments outlined in the paper, please see this other repository: https://github.com/GCrispino/ssp-deadends.

## Installing

1. Create and activate virtual environment:
```
$ python -m venv testenv
$ source testenv/bin/activate
```
2. Install dependencies
```
$ pip install -r requirements.txt
```

## Running

Following is an example of running the solver to solve the Triangle Tireworld environment's second problem (the version defined in the `ipc-envs` branch of a [fork](https://github.com/GCrispino/pddlgym) of the PDDLGym environment), using the algorithm eGUBS-AO*, `0.1` as the value of the `k_g` constant, `-0.3` as the value of the `lambda` risk factor in the eGUBS criterion, and `5` as the number of expansion levels used in eGUBS-AO*:

```
$ python src/main.py --env PDDLEnvTireworld-v0 --problem_index 1 --algorithm_gubs ao --k_g 0.1 --lambda -0.3 --expansion_levels 5
```

Running `$ python src/main.py --help` will print a description of each possible parameter that can be passed to the solver:
```
usage: main.py [-h] --env ENV [--problem_index PROBLEM_INDEX] [--epsilon EPSILON] [--lambda LAMB] [--k_g K_G] [--expansion_levels EXPANSION_LEVELS]
               [--c_max C_MAX] [--algorithm_dc {vi,lao,lao_eliminate_traps,ilao}] [--algorithm_gubs {vi,ao,ao_approx,none}] [--not_p_zero]
               [--eliminate_traps] [--simulate] [--render_and_save] [--output_dir OUTPUT_DIR] [--print_sim_history] [--plot_stats] [--dump_c_maxs]

Solve PDDLGym environments under the GUBS criterion

options:
  -h, --help            show this help message and exit
  --env ENV             PDDLGym environment to solve
  --problem_index PROBLEM_INDEX
                        Chosen environment's problem index to solve (default: 0)
  --epsilon EPSILON     Epsilon used for convergence (default: 0.1)
  --lambda LAMB         Risk factor (default: -0.1)
  --k_g K_G             Constant goal utility (default: -0.1)
  --expansion_levels EXPANSION_LEVELS
                        Expansion levels in eGUBS-AO* (default: 1)
  --c_max C_MAX         C_max value used in approximate version of eGUBS-AO* (default: 50)
  --algorithm_dc {vi,lao,lao_eliminate_traps,ilao}
                        Algorithm to solve the dual criterion (default: vi)
  --algorithm_gubs {vi,ao,ao_approx,none}
                        Algorithm to solve the eGUBS criterion (default: vi)
  --not_p_zero          Defines whether or not to not set probability values to zero in dual criterion value iteration (default: False)
  --eliminate_traps     Defines whether or not to use trap elimination as in FRET (default: False)
  --simulate            Defines whether or not to run a simulation in the problem by applying the algorithm's resulting policy (default: False)
  --render_and_save     Defines whether or not to render and save the received observations during execution to a file (default: False)
  --output_dir OUTPUT_DIR
                        Simulation's output directory (default: ./output)
  --print_sim_history   Defines whether or not to print chosen actions during simulation (default: False)
  --plot_stats          Defines whether or not to run a series of episodes with both a random policy and the policy returned by the algorithm and plot
                        stats about these runs (default: False)
  --dump_c_maxs         Defines whether or not to save cmaxs values for each state on the resulting json output file (default: False)
```

## Testing
Install test dependencies:
```
$ pip install -r test-requirements.txt
```

Then run the following command in your terminal:

```
$ PYTHONPATH=src pytest
```

