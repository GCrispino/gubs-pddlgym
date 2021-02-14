from itertools import chain
import mdp
import utils
from collections import deque
import numpy as np


def shortest_path_from_goal(graph, V_i, s0):
    queue = deque([s0])
    d = {s: 0 for s in graph}
    visited = set([s0])

    while len(queue) > 0:
        s = queue.popleft()
        for s_ in graph[s]:
            if s_ not in graph or s_ in visited:
                continue
            if s_ not in visited:
                queue.append(s_)
                visited.add(s_)
                d[s_] = d[s] + 1

    return d

def tireworld_shortest_path(env):
    problem = env.problems[env._problem_idx]
    road_predicates = utils.get_literals_that_start_with(problem.initial_state, 'road')

    graph = {}
    for road_p in road_predicates:
        origin = road_p.variables[0]
        dest   = road_p.variables[1]

        if dest not in graph:
            #graph[origin] = {'Adj': []}
            graph[dest] = set()
        #graph[origin]['Adj'].append({'state': dest})
        graph[dest].add(origin)
    V_i = {s: i for i, s in enumerate(graph)}
    goal_location = problem.goal.literals[0].variables[0]
    d = shortest_path_from_goal(graph, V_i, goal_location)

    for road_p in road_predicates:
        origin = road_p.variables[0]
        dest   = road_p.variables[1]
        if origin not in graph:
            if origin not in d:
                d[origin] = float('inf')
            d[origin] = min(d[origin], d[dest] + 1)

    return d

def tireworld_h_p_data(env):
    problem = env.problems[env._problem_idx]
    road_predicates = utils.get_literals_that_start_with(problem.initial_state, 'road')

    graph = {}
    for road_p in road_predicates:
        origin = road_p.variables[0]
        dest   = road_p.variables[1]

        if origin not in graph:
            graph[origin] = set()
        graph[origin].add(dest)

    return graph

def tireworld_h_p(env, obs, graph):
    #return 1
    has_flattire = not ('not-flattire()' in set(map(str, obs.literals)))

    location = utils.get_values(obs.literals, 'vehicle-at')[0][0]

    spares = set(chain(*utils.get_values(obs.literals, 'spare-in')))

    if has_flattire and location not in spares:
        return 0

    no_spare_in_succ = True
    for succs in graph[location]:
        for succ in succs:
            if succ in spares:
                no_spare_in_succ = False
    if no_spare_in_succ:
        return 0.5

    return 1


value_heuristic_data_functions = {
    "PDDLEnvTireworld-v0": tireworld_shortest_path
}

prob_heuristic_data_functions = {
    "PDDLEnvTireworld-v0": tireworld_h_p_data
}

def tireworld_h_v(env, obs, lamb, shortest_path):
    #return 1
    has_flattire = not ('not-flattire()' in set(map(str, obs.literals)))

    location = utils.get_values(obs.literals, 'vehicle-at')[0][0]

    spares = set(chain(*utils.get_values(obs.literals, 'spare-in')))

    if has_flattire and location not in spares:
        return 0
    else:
        return np.exp(lamb * shortest_path[location])

value_heuristic_data_functions = {
    "PDDLEnvTireworld-v0": tireworld_shortest_path
}

value_heuristic_functions = {
    "PDDLEnvTireworld-v0": tireworld_h_v
}

prob_heuristic_functions = {
    "PDDLEnvTireworld-v0": tireworld_h_p
}

def build_hv(env, lamb):
    data = None if env.spec.id not in value_heuristic_data_functions else value_heuristic_data_functions[env.spec.id](env)

    def h_v(obs):
        if env.spec.id not in value_heuristic_functions:
            return 1
        return value_heuristic_functions[env.spec.id](env, obs, lamb, data)
    return h_v

def build_hp(env):
    data = None if env.spec.id not in value_heuristic_data_functions else prob_heuristic_data_functions[env.spec.id](env)
    def h_p(obs):
        if env.spec.id not in prob_heuristic_functions:
            return 1
        return prob_heuristic_functions[env.spec.id](env, obs, data)
    return h_p
