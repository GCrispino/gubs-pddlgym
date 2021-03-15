import argparse
import os
import sys
import time
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
import heuristics
import utils

matplotlib.use('TkAgg')

sys.setrecursionlimit(5000)

DEFAULT_PROB_INDEX = 0
DEFAULT_EPSILON = 0.1
DEFAULT_LAMBDA = -0.1
DEFAULT_KG = 1
DEFAULT_ALGORITHM = 'vi'
DEFAULT_NOT_P_ZERO = False
DEFAULT_SIMULATE = False
DEFAULT_RENDER_AND_SAVE = False
DEFAULT_ELIMINATE_TRAPS = False
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
    parser.add_argument('--algorithm_dc',
                        dest='algorithm_dc',
                        choices=['vi', 'lao', 'lao_eliminate_traps', 'ilao'],
                        default=DEFAULT_ALGORITHM,
                        help="Algorithm to solve the dual criterion (default: %s)" % DEFAULT_ALGORITHM)
    parser.add_argument('--algorithm_gubs',
                        dest='algorithm_gubs',
                        choices=['vi', 'ao', 'none'],
                        default=DEFAULT_ALGORITHM,
                        help="Algorithm to solve the eGUBS criterion (default: %s)" % DEFAULT_ALGORITHM)
    parser.add_argument(
        '--not_p_zero',
        dest='not_p_zero',
        default=DEFAULT_NOT_P_ZERO,
        action="store_true",
        help=
        "Defines whether or not to not set probability values to zero in dual criterion value iteration (default: %s)"
        % DEFAULT_NOT_P_ZERO)
    parser.add_argument(
        '--eliminate_traps',
        dest='eliminate_traps',
        default=DEFAULT_ELIMINATE_TRAPS,
        action="store_true",
        help=
        "Defines whether or not to use trap elimination as in FRET (default: %s)"
        % DEFAULT_ELIMINATE_TRAPS)
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
                output_dir=None,
                print_history=False,
                keep_cost=False):
    obs, _ = env.reset()

    render_and_save = output_dir != None
    if render_and_save:
        img = env.render()
        imageio.imsave(os.path.join(output_dir, f"frame1.png"), img)

    cum_reward = 0
    for i in range(1, n_steps + 1):
        old_obs = obs
        a = pi(obs) if not keep_cost else pi(obs, i - 1)
        obs, reward, done, _ = env.step(a)
        cum_reward += reward
        if print_history:
            state_text_render = utils.text_render(env, old_obs)
            if state_text_render:
                print("State:", state_text_render)
            print(" ", pi(old_obs), reward)

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
h_v = heuristics.build_hv(env, args.lamb)
h_p = heuristics.build_hp(env)

A = np.array(
    list(sorted(env.action_space.all_ground_literals(obs, valid_only=False))))
bpsg = None
explicit_graph = None
explicit_graph_dc = None
n_updates = None
n_updates_dc = None
C_max = None
keep_cost = False

print('obtaining optimal policy')
start = time.time()
if args.algorithm_dc == 'vi':
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

    V_dual, P_dual, pi_dual, i_dual = gubs.dual_criterion(args.lamb,
                                                          V_i,
                                                          S,
                                                          h_v,
                                                          goal,
                                                          succ_states,
                                                          A,
                                                          epsilon=args.epsilon)

    n_updates_dc = i_dual * len(S)
    if args.algorithm_gubs == 'vi':
        C_max = gubs.get_cmax(V_dual, V_i, P_dual, S, succ_states, A,
                              args.lamb, args.k_g)
        print("C_max:", C_max)
        V, P, pi = gubs.egubs_vi(V_dual, P_dual, pi_dual, C_max, args.lamb,
                                 args.k_g, V_i, S, goal, succ_states, A)
        pi_func = lambda s, C: pi[V_i[s], C] if C < pi.shape[1] else pi[V_i[s], -1]
        n_updates = (C_max + 1) * len(S)
        keep_cost = True
        print('Result for initial state dc:', P_dual[V_i[obs]], V_dual[V_i[obs]])
        print('Result for initial state:', P[V_i[obs], 0], V[V_i[obs], 0])
    else:
        pi_func = lambda s: pi_dual[V_i[s]]
        n_updates = n_updates_dc
        print('Result for initial state:', P_dual[V_i[obs]], V_dual[V_i[obs]])

elif args.algorithm_dc == 'lao' or args.algorithm_dc == 'lao_eliminate_traps' or args.algorithm_dc == 'ilao':
    if args.algorithm_dc == 'lao_eliminate_traps':
        explicit_graph, bpsg, n_updates = mdp.lao_dual_criterion_fret(
            obs, h_v, h_p, goal, A, args.lamb, env, args.epsilon)
    elif args.algorithm_dc == 'lao':
        explicit_graph, bpsg, n_updates = mdp.lao_dual_criterion(
            obs, h_v, h_p, goal, A, args.lamb, env, args.epsilon, not args.not_p_zero)
    elif args.algorithm_dc == 'ilao':
        explicit_graph, bpsg, n_updates = mdp.ilao_dual_criterion_fret(
            obs, h_v, h_p, goal, A, args.lamb, env, args.epsilon)

    pi_func = lambda s: explicit_graph[s]['pi']

    print('Result for initial state:', explicit_graph[obs]['prob'], explicit_graph[obs]['value'])
elif args.algorithm == 'ao-dualonly-ilao':
    explicit_graph, bpsg, n_updates = mdp.ilao_dual_criterion_fret(
        obs, h_v, h_p, goal, A, args.lamb, env, args.epsilon)

    pi_func = lambda s: explicit_graph[s]['pi']

    print('Result for initial state:', explicit_graph[obs]['prob'], explicit_graph[obs]['value'])
if args.algorithm_gubs == 'ao':
    explicit_graph, bpsg, explicit_graph_dc, C_maxs, n_updates, n_updates_dc, _ = mdp.egubs_ao(
        obs, h_v, h_p, goal, A, args.k_g, args.lamb, env, args.epsilon, args.eliminate_traps)

    C_max = int(C_maxs[obs])
    print("C_max:", C_max)
    solved_states = [s for s, v in explicit_graph_dc.items() if v['solved']]
    pi_func = lambda s, C: explicit_graph[(s, C)]['pi'] if (s, C) in explicit_graph else explicit_graph_dc[s]['pi']
    keep_cost = True
    #print('Size explicit graph dc:', len(explicit_graph_dc))
    print('Result for initial state dc:', explicit_graph_dc[obs]['prob'], explicit_graph_dc[obs]['value'])
    print('Result for initial state:', explicit_graph[(obs, 0)]['prob'], explicit_graph[(obs, 0)]['value'], explicit_graph[(obs, 0)]['value'] + args.k_g * explicit_graph[(obs, 0)]['prob'])
final_time = time.time() - start

if n_updates_dc:
    print("Final updates dc:", n_updates_dc)
print("Final updates:", n_updates)
#if bpsg:
#    res = {(utils.get_coord_from_state(k[0]), k[1]): [ explicit_graph[k]['value'] if k in explicit_graph else None, explicit_graph[k]['prob'] if k in explicit_graph else None, explicit_graph[k]['value'] +
#        args.k_g * explicit_graph[k]['prob'] if k in explicit_graph else None,
#        explicit_graph[k]['pi'] if k in explicit_graph else None
#    ]
#           for k in bpsg}
#
#    print('res: ', res)

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
    plt.plot(range(len(rewards1)),
             np.cumsum(rewards1) / np.arange(1, n_episodes + 1),
             label="optimal")
    plt.plot(range(len(rewards1)),
             np.cumsum(rewards2) / np.arange(1, n_episodes + 1),
             label="random")
    plt.legend()

    plt.figure()
    plt.title('steps')
    plt.plot(range(len(steps1)), np.cumsum(steps1), label="optimal")
    plt.plot(range(len(steps1)), np.cumsum(steps2), label="random")
    plt.legend()
    plt.show()

output_dir = None
# Create folder to save images
if args.render_and_save:
    output_outdir = args.output_dir
    domain_name = env.domain.domain_name
    problem_name = domain_name + str(
        env._problem_idx) if env._problem_index_fixed else None
    output_dir = os.path.join(output_outdir, domain_name, problem_name,
                              f"{str(datetime.now().timestamp())}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

if args.simulate:
    _, goal = run_episode(pi_func,
                          env,
                          n_steps=50,
                          output_dir=output_dir,
                          print_history=args.print_sim_history,
                          keep_cost=keep_cost)
explicit_graph_new_keys = None
if explicit_graph:
    for k, v in explicit_graph.items():
        if 'parents' in v:
            explicit_graph[k]['parents'] = list(explicit_graph[k]['parents'])
    explicit_graph_new_keys = {(str((k[0].literals, k[1])) if type(k) == tuple else str(k.literals)): v
                               for k, v in explicit_graph.items()}

if args.render_and_save:
    output_filename = str(datetime.time(datetime.now())) + '.json'
    output_file_path = utils.output(
        output_filename, {
            **vars(args), 'cpu_time': final_time,
            'n_updates': n_updates,
            'n_updates_dc': n_updates_dc,
            'explicit_graph_size': len(explicit_graph_new_keys) if explicit_graph_new_keys else 0,
            'explicit_graph_dc_size': len(explicit_graph_dc) if explicit_graph_dc else 0,
            'C_max': C_max
        }, output_dir=output_dir)
    if output_file_path:
        print("Algorithm result written to ", output_file_path)
