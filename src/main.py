import argparse
import os
import sys
import gym
import imageio
import pddlgym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from pddlgym.inference import check_goal
import mdp
import gubs

matplotlib.use('TkAgg')

sys.setrecursionlimit(5000)

DEFAULT_PROB_INDEX = 0
DEFAULT_EPSILON = 0.1
DEFAULT_LAMBDA = -0.1
DEFAULT_KG = 1
DEFAULT_ALGORITHM = 'vi'
DEFAULT_SIMULATE = False
DEFAULT_RENDER_AND_SAVE = False
DEFAULT_PRINT_SIM_HISTORY = False
DEFAULT_PLOT_STATS = False
DEFAULT_OUTPUT_DIR = "./output"


def parse_args():
    parser = argparse.ArgumentParser(
        description='Solve PDDLGym environments under the GUBS criterion')

    parser.add_argument('--env',
                        dest='env',
                        required=True,
                        help="PDDLGym environment to solve")
    parser.add_argument(
        '--problem_index',
        type=int,
        default=DEFAULT_PROB_INDEX,
        dest='problem_index',
        help="Chosen environment's problem index to solve (default: %s)" %
        str(DEFAULT_PROB_INDEX))
    parser.add_argument('--epsilon',
                        dest='epsilon',
                        type=float,
                        default=DEFAULT_EPSILON,
                        help="Epsilon used for convergence (default: %s)" %
                        str(DEFAULT_EPSILON))
    parser.add_argument('--lambda',
                        dest='lamb',
                        type=float,
                        default=DEFAULT_LAMBDA,
                        help="Risk factor (default: %s)" % str(DEFAULT_LAMBDA))
    parser.add_argument('--k_g',
                        dest='k_g',
                        type=float,
                        default=DEFAULT_KG,
                        help="Constant goal utility (default: %s)" %
                        str(DEFAULT_LAMBDA))
    parser.add_argument('--algorithm', dest='algorithm', choices=['vi', 'vi-dualonly', 'ao-dualonly'],
                        default=DEFAULT_ALGORITHM,
                        help="Algorithm to run (default: %s)" % DEFAULT_ALGORITHM)
    parser.add_argument(
        '--simulate',
        dest='simulate',
        default=DEFAULT_SIMULATE,
        action="store_true",
        help=
        "Defines whether or not to run a simulation in the problem by applying the algorithm's resulting policy (default: %s)"
        % DEFAULT_SIMULATE)
    parser.add_argument(
        '--render_and_save',
        dest='render_and_save',
        default=DEFAULT_RENDER_AND_SAVE,
        action="store_true",
        help=
        "Defines whether or not to render and save the received observations during execution to a file (default: %s)"
        % DEFAULT_RENDER_AND_SAVE)
    parser.add_argument('--output_dir',
                        dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help="Simulation's output directory (default: %s)" %
                        DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        '--print_sim_history',
        dest='print_sim_history',
        action="store_true",
        default=DEFAULT_PRINT_SIM_HISTORY,
        help=
        "Defines whether or not to print chosen actions during simulation (default: %s)"
        % DEFAULT_PRINT_SIM_HISTORY)

    parser.add_argument(
        '--plot_stats',
        dest='plot_stats',
        action="store_true",
        default=DEFAULT_PLOT_STATS,
        help=
        "Defines whether or not to run a series of episodes with both a random policy and the policy returned by the algorithm and plot stats about these runs (default: %s)"
        % DEFAULT_PLOT_STATS)

    return parser.parse_args()


def run_episode(pi,
                env,
                n_steps=50,
                render_and_save=False,
                output_dir=".",
                print_history=False,
                keep_cost=False):
    obs, _ = env.reset()

    # Create folder to save images
    if render_and_save:
        output_outdir = args.output_dir
        domain_name = env.domain.domain_name
        problem_name = domain_name + str(
            env._problem_idx) if env._problem_index_fixed else None
        output_dir = os.path.join(output_outdir, domain_name, problem_name,
                                  f"{str(datetime.now().timestamp())}")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    if render_and_save:
        img = env.render()
        imageio.imsave(os.path.join(output_dir, f"frame1.png"), img)

    cum_reward = 0
    for i in range(1, n_steps + 1):
        old_obs = obs
        obs, reward, done, _ = env.step(
            pi(obs) if not keep_cost else pi(obs, i - 1))
        cum_reward += reward
        if print_history:
            print(pi(old_obs), reward)

        if render_and_save:
            img = env.render()
            imageio.imsave(os.path.join(output_dir, f"frame{i + 1}.png"), img)

        if done:
            break
    return i, cum_reward

def h_1(_):
    return 1

args = parse_args()

env = gym.make(args.env)
problem_index = args.problem_index
env.fix_problem_index(problem_index)
problem = env.problems[problem_index]
goal = problem.goal
prob_objects = frozenset(problem.objects)

obs, _ = env.reset()
A = np.array(list(sorted(env.action_space.all_ground_literals(obs, valid_only=False))))


print('obtaining optimal policy')
if args.algorithm == 'vi-dualonly' or args.algorithm == 'vi':
    print(' calculating list of states...')
    reach = mdp.get_all_reachable(obs, A, env)
    S = list(sorted([s for s in reach]))
    print('Number of states:', len(S))

    print('done')
    V_i = {s: i for i, s in enumerate(S)}
    G_i = [V_i[s] for s in V_i if check_goal(s, goal)]
    #
    succ_states = {s: {} for s in reach}
    for s in reach:
        for a in A:
            succ_states[s, a] = reach[s][a]

    V_dual, P_dual, pi_dual, i_dual = gubs.dual_criterion(args.lamb, V_i,
                                                          S,
                                                          goal,
                                                          succ_states,
                                                          A,
                                                          epsilon=args.epsilon)

    if args.algorithm == 'vi':
        C_max = gubs.get_cmax(V_dual, V_i, P_dual, S, succ_states, A, args.lamb,
                              args.k_g)
        V, P, pi = gubs.egubs_vi(V_dual, P_dual, pi_dual, C_max, args.lamb, args.k_g,
                                 V_i, S, goal, succ_states, A)
        pi_func = lambda s, C: pi[V_i[s], C] if C < pi.shape[1] else pi[V_i[s], -1]
    else:
        pi_func = lambda s: pi_dual[V_i[s]]
        print(obs, V_dual[V_i[obs]], P_dual[V_i[obs]], pi_dual[V_i[obs]])
elif args.algorithm == 'ao-dualonly':
    explicit_graph, bpsg, n_iter = mdp.lao_dual_criterion(obs, h_1, h_1, goal, A, args.lamb, env, args.epsilon)

    pi_func = lambda s: explicit_graph[s]['pi']
    print(obs, explicit_graph[obs]['value'], explicit_graph[obs]['prob'], explicit_graph[obs]['pi'])

#V = np.zeros(len(explicit_graph))
#P = np.zeros(len(explicit_graph))
#pi = np.full(len(explicit_graph), None)
#state_coords = {}
#nx = 0
#ny = 0
#for s, v in explicit_graph.items():
#    state_name = [lit for lit in s.literals if lit.predicate.name.startswith('robot-at')][0].variables[1].name
#    state_x, state_y = [int(s) for s in state_name[1:-1].split('-')]
#    state_coords[s] = (state_x, state_y)
#    nx = max(nx, state_x + 1)
#    ny = max(ny, state_y + 1)
#
#print(nx, ny)
#for s, (x, y) in state_coords.items():
#    state_i = y * nx + x
#    print((x, y), state_i)
#    V[state_i] = explicit_graph[s]['value']
#    P[state_i] = explicit_graph[s]['prob']
#    pi[state_i] = explicit_graph[s]['pi']
#
#print(V.reshape(ny, nx))
#print(P.reshape(ny, nx))
#print(pi.reshape(ny, nx))


n_episodes = 500

if args.plot_stats:
    print('running episodes with optimal policy')
    steps1 = []
    rewards1 = []
    for i in range(n_episodes):
        n_steps, reward = run_episode(pi_func, env, keep_cost=False)
        steps1.append(n_steps)
        rewards1.append(reward)

    print('running episodes with random policy')
    steps2 = []
    rewards2 = []
    for i in range(n_episodes):
        n_steps, reward = run_episode(lambda s: env.action_space.sample(s),
                                      env)
        steps2.append(n_steps)
        rewards2.append(reward)
    rewards2 = np.array(rewards2)

    plt.title('Cumulative reward')
    plt.plot(range(len(rewards1)), np.cumsum(rewards1), label="optimal")
    plt.plot(range(len(rewards1)), np.cumsum(rewards2), label="random")
    plt.legend()

    plt.figure()
    plt.title('Average reward')
    plt.plot(range(len(rewards1)), np.cumsum(rewards1) / np.arange(1, n_episodes + 1), label="optimal")
    plt.plot(range(len(rewards1)), np.cumsum(rewards2) / np.arange(1, n_episodes + 1), label="random")
    plt.legend()

    plt.figure()
    plt.title('steps')
    plt.plot(range(len(steps1)), np.cumsum(steps1), label="optimal")
    plt.plot(range(len(steps1)), np.cumsum(steps2), label="random")
    plt.legend()
    plt.show()

if args.simulate:
    _, goal = run_episode(pi_func,
                          env,
                          n_steps=50,
                          render_and_save=args.render_and_save,
                          output_dir=args.output_dir,
                          print_history=args.print_sim_history,
                          keep_cost=False)
