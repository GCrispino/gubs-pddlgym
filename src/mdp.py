from copy import copy
import gubs
from utils import text_render, flatten
import numpy as np
from pddlgym.core import get_successor_states, InvalidAction
from pddlgym.inference import check_goal

def add_state_graph(s, graph, to_str=False, add_expanded_prop=False):
    graph_ = copy(graph)
    graph_[str(s) if to_str else s] = {'Adj': []}

    return graph_

def get_successor_states_check_exception(s, a, domain, return_probs=True):
    try:
        succ = get_successor_states(s,
                                    a,
                                    domain,
                                    raise_error_on_invalid_action=True,
                                    return_probs=return_probs)
    except InvalidAction:
        succ = {s: 1.0} if return_probs else frozenset({s})

    return succ


def get_all_reachable(s, A, env, reach=None):
    reach = {} if not reach else reach

    reach[s] = {}
    for a in A:
        succ = get_successor_states_check_exception(s, a, env.domain)

        reach[s][a] = {s_: prob for s_, prob in succ.items()}
        for s_ in succ:
            if s_ not in reach:
                reach.update(get_all_reachable(s_, A, env, reach))
    return reach


def vi(S, succ_states, A, V_i, G_i, goal, env, gamma, epsilon):

    V = np.zeros(len(V_i))
    P = np.zeros(len(V_i))
    pi = np.full(len(V_i), None)
    print(len(S), len(V_i), len(G_i), len(P))
    print(G_i)
    P[G_i] = 1

    i = 0
    diff = np.inf
    while True:
        print('Iteration', i, diff)
        V_ = np.copy(V)
        P_ = np.copy(P)

        for s in S:
            if check_goal(s, goal):
                continue
            Q = np.zeros(len(A))
            Q_p = np.zeros(len(A))
            cost = 1
            for i_a, a in enumerate(A):
                succ = succ_states[s, a]

                probs = np.fromiter(iter(succ.values()), dtype=float)
                succ_i = [V_i[succ_s] for succ_s in succ_states[s, a]]
                Q[i_a] = cost + np.dot(probs, gamma * V_[succ_i])
                Q_p[i_a] = np.dot(probs, P_[succ_i])
            V[V_i[s]] = np.min(Q)
            P[V_i[s]] = np.max(Q_p)
            pi[V_i[s]] = A[np.argmin(Q)]

        diff = np.linalg.norm(V_ - V, np.inf)
        if diff < epsilon:
            break
        i += 1
    return V, pi


def expand_state_dual_criterion(s, h_v, h_p, env, explicit_graph, goal, A):
    if check_goal(s, goal):
        raise ValueError(
            f'State {s} can\'t be expanded because it is a goal state')

    neighbour_states_dict = {}
    neighbour_states = []
    i = 0
    for a in A:
        succs = get_successor_states_check_exception(s, a, env.domain)
        for s_, p in succs.items():
            if s_ not in neighbour_states_dict:
                neighbour_states_dict[s_] = i
                i += 1
                neighbour_states.append({'state': s_, 'A': {a: p}})
            else:
                neighbour_states[neighbour_states_dict[s_]]['A'][a] = p

    unexpanded_neighbours = filter(
        lambda _s: (not _s['state'] in explicit_graph) or (not explicit_graph[_s['state']]['expanded']), neighbour_states)

    # Add new empty states to 's' adjacency list
    new_explicit_graph = copy(explicit_graph)

    new_explicit_graph[s]['prob'] = 0

    new_explicit_graph[s]["Adj"].extend(neighbour_states)

    for n in unexpanded_neighbours:
        if n['state'] != s:
            is_goal = check_goal(n['state'], goal)
            new_explicit_graph[n['state']] = {
                "value": 1 if is_goal else h_v(n['state']),
                "prob": 1 if is_goal else h_p(n['state']),
                "solved": False,
                "pi": None,
                "expanded": False,
                "Adj": []
            }

    new_explicit_graph[s]['expanded'] = True

    return new_explicit_graph

def expand_state_gubs(s, env, goal, explicit_graph, C, C_maxs, V_risk, P_risk, pi_risk, V_i, A):
    if check_goal(s[0], goal):
        raise ValueError(
            f'State {s[0]} can\'t be expanded because it is a goal state')

    # Get 's' neighbour states that were not expanded
    #    and "collapse" equal (s, C) pairs
    neighbour_states_dict = {}
    neighbour_states = []

    i = 0
    for a in A:
        succs = get_successor_states_check_exception(s[0], a, env.domain)
        for s_, p in succs.items():
            c_ = C(s_, a)
            if (s_, c_) not in neighbour_states_dict:
                neighbour_states_dict[(s_, c_)] = i
                i += 1
                neighbour_states.append({'state': (s_, c_), 'A': {a: p}})
            else:
                neighbour_states[neighbour_states_dict[(s_, c_)]]['A'][a] = p

    unexpanded_neighbours = list(filter(
        lambda _s: ((_s['state'] not in explicit_graph) or (not explicit_graph[_s['state']]['expanded'])), neighbour_states))

    # Add new empty states to 's' adjacency list
    new_explicit_graph = copy(explicit_graph)
    new_explicit_graph[s]["Adj"].extend(neighbour_states)

    #new_mdp = deepcopy(mdp)
    for neigh in unexpanded_neighbours:
        n = neigh['state']
        print(n, type(n))
        # Check if maximum cost for state was reached
        c_max_n = max([C_maxs[n[0]][a]
                       for a in C_maxs[n[0]]]) if n[0] in C_maxs else 0

        solved = n[1] >= c_max_n
        value = V_risk[V_i[n[0]]] if solved else 1
        prob = P_risk[V_i[n[0]]]
        new_explicit_graph[n] = {
            "solved": solved,
            "value": value,
            "prob": prob,
            "pi": pi_risk[V_i[n[0]]],
            "expanded": False,
            "Adj": []
        }

    new_explicit_graph[s]['expanded'] = True

    return new_explicit_graph

def expand_state_gubs_v2(s, h, env, goal, explicit_graph, C, C_maxs, V_risk, P_risk, pi_risk, V_i, A, k_g, lamb):
    if check_goal(s[0], goal):
        raise ValueError(
            f'State {s[0]} can\'t be expanded because it is a goal state')

    # Get 's' neighbour states that were not expanded
    #    and "collapse" equal (s, C) pairs
    neighbour_states_dict = {}
    neighbour_states = []

    i = 0
    for a in A:
        succs = get_successor_states_check_exception(s[0], a, env.domain)
        for s_, p in succs.items():
            c_ = s[1] + C(s_, a)
            if (s_, c_) not in neighbour_states_dict:
                neighbour_states_dict[(s_, c_)] = i
                i += 1
                neighbour_states.append({'state': (s_, c_), 'A': {a: p}})
            else:
                neighbour_states[neighbour_states_dict[(s_, c_)]]['A'][a] = p


    unexpanded_neighbours = list(filter(
        lambda _s: ((_s['state'] not in explicit_graph) or (not explicit_graph[_s['state']]['expanded'])), neighbour_states))
    expanded_neighbours = list(filter(
        lambda _s: ((_s['state'] in explicit_graph) and explicit_graph[_s['state']]['expanded']), neighbour_states))

    # Add new empty states to 's' adjacency list
    new_explicit_graph = copy(explicit_graph)

    new_explicit_graph[s]["Adj"].extend(neighbour_states)

    for neigh in expanded_neighbours:
        n = neigh['state']
        if s not in new_explicit_graph[n]['parents']:
            new_explicit_graph[n]['parents'].add(s)

    for neigh in unexpanded_neighbours:
        n = neigh['state']
        # Check if maximum cost for state is known and was reached
        if n[0] not in C_maxs:
            raise NotImplementedError()
            # Compute succ states
            #C_maxs = gubs.get_cmax_reachable(
            #    n[0], V_risk, V_i, P_risk, pi_risk, goal, A, C, lamb, k_g, succ_states, W_s=C_maxs)
        else:
            c_max_n = C_maxs[n[0]]

        solved = bool(check_goal(n[0], goal) or n[1] >= c_max_n)
        value = V_risk[V_i[n[0]]] if solved else h(n[0])
        prob = P_risk[V_i[n[0]]]
        pi = pi_risk[n[0]] if solved else None
        if n in new_explicit_graph:
            new_explicit_graph[n]['parents'].add(s)
        else:
            new_explicit_graph[n] = {
                "solved": solved,
                "value": value,
                "prob": prob,
                "pi": pi,
                "expanded": False,
                "Adj": [],
                "parents": set([s])
            }

    new_explicit_graph[s]['expanded'] = True

    return new_explicit_graph, C_maxs


def get_unexpanded_states(goal, explicit_graph, bpsg):
    return list(
        filter(
            lambda x: (x not in explicit_graph) or (not explicit_graph[x]["expanded"] and not check_goal(
                x, goal)), bpsg.keys()))

def get_unexpanded_states_extended(goal, explicit_graph, bpsg):
    return list(
        filter(lambda x: x in explicit_graph and not explicit_graph[x]['expanded'] and not explicit_graph[x]["solved"]
               and not check_goal(x[0], goal), bpsg.keys())
    )

def find_reachable(s, a, mdp):
    """ Find states that are reachable from state 's' after executing action 'a' """
    all_reachable_from_s = mdp[s]['Adj']
    return list(filter(
        lambda obj_s_: a in obj_s_['A'],
        all_reachable_from_s
    ))



def find_direct_ancestors(s, graph, visited, best=False):
    return list(
        filter(
            lambda s_: s_ != s and (s_ not in visited) and len(
                list(
                    filter(
                        lambda s__: s__['state'] == s and
                        (True if not best else graph[s_]['pi'] in s__['A']
                         ), graph[s_]['Adj']))) > 0, graph))


def __find_ancestors(s, bpsg, visited, best):
    # Find states in graph that have 's' in the adjacent list (except from 's' itself and states that were already visited):
    direct_ancestors = list(
        filter(lambda a: a not in visited,
               find_direct_ancestors(s, bpsg, visited, best)))

    result = [] + direct_ancestors

    for a in direct_ancestors:
        if a not in visited:
            visited.add(a)
            result += __find_ancestors(a, bpsg, visited, best)

    return result


def find_ancestors(s, bpsg, best=False):
    return __find_ancestors(s, bpsg, set(), best)

def find_direct_ancestors_extended(s, graph, best=False):
    return list(filter(
        lambda s_: s_ != s and len(
            list(filter(
                lambda s__: s__['state'] == s and (
                    True
                    if not best
                    else graph[s_]['pi'] in s__['A']
                ),
                graph[s_]['Adj']
            ))
        ) > 0,
        graph[s]['parents'] if 'parents' in graph[s] else graph
    ))


def __find_ancestors_extended(s, bpsg, visited, C, best=False, sorting=None):
    sorting = [] if sorting == None else sorting
    direct_ancestors = list(filter(
        lambda a: a not in visited,
        find_direct_ancestors_extended(s, bpsg, best)
    ))

    result = [] + direct_ancestors

    for a in direct_ancestors:
        if a not in visited:
            visited.add(a)
            result_, _ = __find_ancestors_extended(
                a, bpsg, visited, C, best, sorting)
            result += result_
    sorting.append(s)

    return result, sorting


def find_ancestors_extended(s, bpsg, C, best=False):
    return __find_ancestors_extended(s, bpsg, set(), C, best)


def find_neighbours(s, adjs):
    """ Find neighbours of s in adjacency list (except itself) """
    return list(
        map(lambda s_: s_['state'],
            filter(lambda s_: s_['state'] != s, adjs)))

def find_unreachable(s0, mdp):
    """ search for unreachable states using dfs """
    S = list(mdp.keys())
    len_s = len(S)
    V_i = {S[i]: i for i in range(len_s)}
    colors = ['w'] * len_s
    dfs_visit(V_i[s0], colors, [-1] * len_s,
              [-1] * len_s, [0], S, V_i, mdp)
    return [S[i] for i, c in enumerate(colors) if c != 'b']


def dfs_visit(i, colors, d, f, time, S, V_i, mdp, fn=None):
    colors[i] = 'g'
    time[0] += 1
    d[i] = time[0]
    s = S[i]

    for s_obj in mdp[s]['Adj']:
        s_ = s_obj['state']
        if s_ not in mdp:
            continue
        j = V_i[s_]
        if colors[j] == 'w':
            dfs_visit(j, colors, d, f, time, S, V_i, mdp, fn)

    if fn:
        fn(s)

    colors[i] = 'b'
    time[0] += 1
    f[i] = time[0]


def dfs(mdp, fn=None):
    S = list(mdp.keys())
    len_s = len(S)
    V_i = {S[i]: i for i in range(len_s)}
    # (w)hite, (g)ray or (b)lack
    colors = ['w'] * len_s
    d = [-1] * len_s
    f = [-1] * len_s
    time = [0]
    for i in range(len_s):
        c = colors[i]
        if c == 'w':
            dfs_visit(i, colors, d, f, time, S, V_i, mdp, fn)

    return d, f, colors

def topological_sort(mdp):
    stack = []

    def dfs_fn(s):
        stack.append(s)
    dfs(mdp, dfs_fn)
    return list(reversed(stack))

def update_action_partial_solution(s, s0, a, bpsg, explicit_graph):
    """
        Updates partial solution given pair of state and action
    """
    bpsg_ = copy(bpsg)
    i = 0
    states = [s]
    while len(states) > 0:
        s = states.pop()
        a = explicit_graph[s]['pi']
        s_obj = bpsg_[s]

        s_obj['Adj'] = []
        reachable = find_reachable(s, a, explicit_graph)

        for s_obj_ in reachable:
            s_ = s_obj_['state']
            s_obj['Adj'].append({'state': s_, 'A': {a: s_obj_['A'][a]}})
            if s_ not in bpsg_:
                bpsg_ = add_state_graph(s_, bpsg_)
                bpsg_[s] = s_obj

                if explicit_graph[s_]['expanded']:
                    states.append(s_)
        i += 1

    unreachable = find_unreachable(s0, bpsg_)

    for s_ in unreachable:
        if s_ in bpsg_:
            bpsg_.pop(s_)

    return bpsg_


def update_partial_solution(s0, bpsg, explicit_graph):
    bpsg_ = copy(bpsg)
    S = explicit_graph.keys()

    for s in S:
        a = explicit_graph[s]['pi']
        if s not in bpsg_:
            continue

        s_obj = bpsg_[s]

        if len(s_obj['Adj']) == 0:
            if a is not None:
                bpsg_ = update_action_partial_solution(s, s0, a, bpsg_,
                                                       explicit_graph)
        else:
            best_current_action = next(iter(s_obj['Adj'][0]['A'].keys()))

            if a is not None and best_current_action != a:
                bpsg_ = update_action_partial_solution(s, s0, a, bpsg_,
                                                       explicit_graph)

    return bpsg_

def update_action_partial_solution_extended(s, s0, C, a, bpsg, explicit_graph):
    """
        Updates partial solution given pair of state and action
    """
    bpsg_ = copy(bpsg)
    i = 0
    states = [s]
    while len(states) > 0:
        s = states.pop()
        # a = pi[V_i[s[0]], s[1]]
        a = explicit_graph[s]['pi']
        s_obj = bpsg_[s]

        s_obj['Adj'] = []
        #reachable = find_reachable(s[0], a, mdp)
        reachable = find_reachable(s, a, explicit_graph)

        for s_obj_ in reachable:
            s_ = s_obj_['state']
            #c_ = s[1] + C(s[0], a)
            #s_extended = (s_, c_)
            s_obj['Adj'].append({
                'state': s_,
                'A': {a: s_obj_['A'][a]}
            })

            #if s_extended not in bpsg_:
            if s_ not in bpsg_:

                #bpsg_[s_extended] = {
                bpsg_[s_] = {
                    "Adj": []
                }
                bpsg_[s] = s_obj

                #if s_extended in explicit_graph and explicit_graph[s_extended]['expanded']:
                if s_ in explicit_graph and explicit_graph[s_]['expanded']:
                    states.append(s_)
        i += 1

    unreachable = find_unreachable((s0, 0), bpsg_)

    for s_ in unreachable:
        if s_ in bpsg_:
            bpsg_.pop(s_)

    return bpsg_


def update_partial_solution_extended(s0, C, bpsg, explicit_graph):
    bpsg_ = copy(bpsg)
    S = explicit_graph.keys()

    for s in S:
        a = explicit_graph[s]['pi']
        if s not in bpsg_:
            continue

        s_obj = bpsg_[s]

        if len(s_obj['Adj']) == 0:
            if a is not None:
                bpsg_ = update_action_partial_solution_extended(
                    s, s0, C, a, bpsg_, explicit_graph)
        else:
            best_current_action = next(iter(s_obj['Adj'][0]['A'].keys()))

            if a is not None and best_current_action != a:
                bpsg_ = update_action_partial_solution_extended(
                    s, s0, C, a, bpsg_, explicit_graph)

    return bpsg_


def value_iteration_dual_criterion(explicit_graph,
                                   bpsg,
                                   A,
                                   Z,
                                   goal,
                                   lamb,
                                   C,
                                   epsilon=1e-3,
                                   n_iter=None,
                                   convergence_test=False):
    n_states = len(explicit_graph)

    # initialize
    V = np.zeros(n_states, dtype=float)
    pi = np.full(n_states, None)
    P = np.zeros(n_states, dtype=float)
    V_i = {s: i for i, s in enumerate(explicit_graph)}

    for s in Z:
        if not check_goal(s, goal) and not explicit_graph[s]['solved'] and explicit_graph[s]['expanded']:
            explicit_graph[s]['prob'] = 0

    for s, n in explicit_graph.items():
        V[V_i[s]] = n['value']
        P[V_i[s]] = n['prob']
        pi[V_i[s]] = n['pi']

    i = 0

    P_not_max_prob = np.copy(P)
    V_ = np.copy(V)
    P_ = np.copy(P)
    pi_ = np.copy(pi)

    changed = False
    converged = False
    n_updates = 0
    np.seterr(all='raise')
    while True:
        for s in Z:
            if explicit_graph[s]['solved']:
                continue
            n_updates += 1
            all_reachable = np.array(
                [find_reachable(s, a, explicit_graph) for a in A], dtype=object)
            actions_results_p = np.array([
                np.sum([
                    P[V_i[s_['state']]] * s_['A'][a] for s_ in all_reachable[i]
                ]) for i, a in enumerate(A)
            ])

            # set maxprob
            max_prob = np.max(actions_results_p)
            P_[V_i[s]] = max_prob
            i_A_max_prob = np.argwhere(
                actions_results_p == max_prob).reshape(-1)
            A_max_prob = A[i_A_max_prob]
            not_max_prob_actions_results = actions_results_p[
                actions_results_p != max_prob]

            P_not_max_prob[V_i[s]] = P[V_i[s]] if len(
                not_max_prob_actions_results) == 0 else np.max(
                    not_max_prob_actions_results)

            actions_results = np.array([
                np.sum([
                    np.exp(lamb * C(s, A[i])) * V[V_i[s_['state']]] *
                    s_['A'][A[i]] for s_ in all_reachable[i]
                ]) for i in i_A_max_prob
            ])

            i_a = np.argmax(actions_results)
            V_[V_i[s]] = actions_results[i_a]
            pi_[V_i[s]] = A_max_prob[i_a]

        #print(V_i.values())
        #print(len(V_i))
        v_norm = np.linalg.norm(V_[list(V_i.values())] - V[list(V_i.values())],
                                np.inf)
        p_norm = np.linalg.norm(P_[list(V_i.values())] - P[list(V_i.values())],
                                np.inf)

        P_diff = P_[list(V_i.values())] - P_not_max_prob[list(V_i.values())]
        arg_min_p_diff = np.argmin(P_diff)
        min_p_diff = P_diff[arg_min_p_diff]

        different_actions = pi_[list(
            V_i.values())][pi_[list(V_i.values())] != pi[list(V_i.values())]]
        if len(different_actions) > 0:
            changed = True
        if v_norm + p_norm < epsilon and min_p_diff >= 0:
            converged = True
        V = np.copy(V_)
        P = np.copy(P_)
        pi = np.copy(pi_)

        # if (convergence_test and changed) or converged:
        if converged:
            print(
                f'{convergence_test}, {changed}, {converged}, {v_norm}, {p_norm}, {min_p_diff}'
            )
            break

        i += 1

    # save results in explicit graph
    for s in Z:
        explicit_graph[s]['value'] = V[V_i[s]]
        explicit_graph[s]['prob'] = P[V_i[s]]
        explicit_graph[s]['pi'] = pi[V_i[s]]

    #print(f'{i} iterations')
    return explicit_graph, converged, n_updates


def lao_dual_criterion(s0,
                       h_v,
                       h_p,
                       goal,
                       A,
                       lamb,
                       env,
                       epsilon=1e-3,
                       explicit_graph=None):
    bpsg = {s0: {"Adj": []}}
    explicit_graph = explicit_graph or {}

    if s0 in explicit_graph and explicit_graph[s0]['solved']:
        return explicit_graph, bpsg, 0

    if s0 not in explicit_graph:
        explicit_graph[s0] = {
            "value": h_v(s0),
            "prob": 0,
            "solved": False,
            "expanded": False,
            "pi": None,
            "Adj": []
        }

    def C(s, a):
        return 0 if check_goal(s, goal) else 1

    i = 1
    #unexpanded = [s0]
    unexpanded = get_unexpanded_states(goal, explicit_graph, bpsg)
    n_updates = 0
    explicit_graph_cur_size = 1
    while True:
        while len(unexpanded) > 0:
            s = unexpanded[0]
            print("Iteration", i)
            #print("Will expand", text_render(env, s))
            unexpanded = unexpanded
            print("Will expand", len(unexpanded), "states")
            Z = set()
            for s in unexpanded:
                explicit_graph = expand_state_dual_criterion(
                    s, h_v, h_p, env, explicit_graph, goal, A)
                Z.add(s)
                Z.update(find_ancestors(s, explicit_graph, best=True))
            #Z = [s] + find_ancestors(s, explicit_graph, best=True)
            assert len(explicit_graph) >= explicit_graph_cur_size
            explicit_graph_cur_size = len(explicit_graph)
            print("explicit graph size:", explicit_graph_cur_size)
            print("Z size:", len(Z))
            explicit_graph, _, n_updates_ = value_iteration_dual_criterion(
                explicit_graph, bpsg, A, Z, goal, lamb, C, epsilon=epsilon)
            print(f"Finished value iteration in {n_updates_} updates")
            n_updates += n_updates_
            bpsg = update_partial_solution(s0, bpsg, explicit_graph)
            unexpanded = get_unexpanded_states(goal, explicit_graph, bpsg)
            i += 1
        bpsg_states = [s_ for s_ in bpsg.keys() if not check_goal(s_, goal)]
        explicit_graph, converged, n_updates_ = value_iteration_dual_criterion(
            explicit_graph,
            bpsg,
            A,
            bpsg_states,
            goal,
            lamb,
            C,
            epsilon=epsilon,
            convergence_test=True)
        n_updates += n_updates_

        bpsg = update_partial_solution(s0, bpsg, explicit_graph)
        unexpanded = get_unexpanded_states(goal, explicit_graph, bpsg)

        if converged and len(unexpanded) == 0:
            break
    for s_ in bpsg:
        explicit_graph[s_]['solved'] = True
    return explicit_graph, bpsg, n_updates

def lao_dual_criterion_reachable(s0, h_v, h_p, goal, A, lamb, env, epsilon=1e-3):
    #all_reachable = mg.find_all_reachable(s0, mdp_obj)
    all_reachable = set({s0})
    explicit_graph = {}

    stack = [s0]
    n_updates_total = 0
    while len(stack) > 0:
        s = stack.pop()
        print("Will call lao_dual_criterion for state:", text_render(env, s))
        explicit_graph, _, n_updates = lao_dual_criterion(
            s, h_v, h_p, goal, A, lamb, env, epsilon=epsilon, explicit_graph=explicit_graph)
        n_updates_total += n_updates
        print(' finished lao dual criterion for', s, explicit_graph[s]['prob'], explicit_graph[s]['value'], len(
            [v for v in explicit_graph.values() if v['solved']]))
        for s_ in explicit_graph:
            if s_ not in all_reachable:
                all_reachable.add(s_)
                stack.append(s_)

    #pi = {s: explicit_graph[s]['pi']
    #      for s in sorted(explicit_graph, key=int)}
    #print(' pi:', pi)
    return explicit_graph, n_updates_total

def value_iteration_gubs(explicit_graph, V_i, A, Z, k_g, lamb, C, env):
    n_actions = len(A)
    changed = False
    stack = copy(Z)
    z_graph = {s: {'Adj': []} for s in stack}

    i = 0
    while len(stack) > 0:
        s = stack.pop(0)
        z_graph.pop(s)

        if explicit_graph[s]['solved']:
            print("SOLVED", s)
            continue
        c = s[1]
        q_actions_results = np.zeros(n_actions)
        p_actions_results = np.zeros(n_actions)
        all_reachable = []
        for i_a, a in enumerate(A):
            c_s_a = C(s[0], a)
            #c_ = c + c_s_a
            #reachable = find_reachable(s[0], a, mdp_obj)
            reachable = find_reachable(s, a, explicit_graph)
            all_reachable.append(reachable)

            # Get value
            #gen_q = (s_['A'][a] * explicit_graph[(s_['name'], c_)]['value']
                     #for s_ in reachable)
            gen_q = (s_['A'][a] * explicit_graph[s_['state']]['value']
                     for s_ in reachable)
            q_actions_results[i_a] = np.exp(lamb * c_s_a) * \
                np.sum(np.fromiter(gen_q, dtype=np.float))

            # Get probability
            #gen_p = (s_['A'][a] * explicit_graph[(s_['name'], c_)]['prob']
            #         for s_ in reachable)
            gen_p = (s_['A'][a] * explicit_graph[s_['state']]['prob']
                     for s_ in reachable)
            p_actions_results[i_a] = np.sum(
                np.fromiter(gen_p, dtype=np.float)
            )

        i_a_opt = np.argmax(
            np.exp(lamb * c) * q_actions_results + k_g * p_actions_results
        )

        reachable_opt = all_reachable[i_a_opt]
        c_s_a = C(s[0], A[i_a_opt])
        c_ = c + c_s_a
        #is_solved = all([explicit_graph[(r['name'], c_)]['solved']
        #                 for r in reachable_opt])
        is_solved = all([explicit_graph[r['state']]['solved']
                         for r in reachable_opt])

        old_val = explicit_graph[s]['value']
        if explicit_graph[s]['value'] != q_actions_results[i_a_opt]:
            explicit_graph[s]['value'] = q_actions_results[i_a_opt]
            changed = True
        if explicit_graph[s]['prob'] != p_actions_results[i_a_opt]:
            explicit_graph[s]['prob'] = p_actions_results[i_a_opt]
            changed = True
        if explicit_graph[s]['pi'] != A[i_a_opt]:
            explicit_graph[s]['pi'] = A[i_a_opt]
            changed = True

        if is_solved:
            print(f"{text_render(env, s[0])} with cost {s[1]} is now solved!")
        explicit_graph[s]['solved'] = is_solved

        if explicit_graph[s]['value'] < old_val or is_solved:
            # Label as solved and add ancestors to stack
            ancestors = [a for a in find_direct_ancestors_extended(
                s, explicit_graph, best=True) if a not in stack]
            if len(ancestors) > 0:
                # print(
                #     f"Mudou no estado {s}! Adiciona os seguintes à lista:", ancestors, [explicit_graph[a]['pi'] for a in ancestors])
                s_adj_obj = {'state': s}
                for a in ancestors:
                    if a not in z_graph:
                        z_graph[a] = {}
                    if 'Adj' not in z_graph[a]:
                        z_graph[a]['Adj'] = []
                    z_graph[a]['Adj'].append(copy(s_adj_obj))
                stack = list(reversed(topological_sort(z_graph)))
                # print('  Nova lista:', stack)

        i += 1
    return explicit_graph, i, changed

def egubs_ao(s0, h_v, h_p, goal, A, k_g, lamb, env, epsilon=1e-3):
    explicit_graph_dc, n_updates_dc = lao_dual_criterion_reachable(
        s0, h_v, h_p, goal, A, lamb, env, epsilon)

    V_risk = {s: explicit_graph_dc[s]['value']
              for s in explicit_graph_dc}
    P_risk = {s: explicit_graph_dc[s]['prob']
              for s in explicit_graph_dc}
    pi_risk = {s: explicit_graph_dc[s]['pi']
               for s in explicit_graph_dc}
    V_i = {s: s for s in V_risk}

    def C(s, a): return 0 if check_goal(s, goal) else 1

    # Criar succ_states a partir do explicit_graph_dc
    # TODO -> Imprimir esse cara pra uma instância do rio e ver se ta certo
    succ_states = {}
    for s_ in explicit_graph_dc:
        for s__ in explicit_graph_dc[s_]['Adj']:
            for a, p in s__['A'].items():
                if (s_, a) not in succ_states:
                    succ_states[s_, a] = {}
                if s__['state'] not in succ_states[s_, a]:
                    succ_states[s_, a][s__['state']] = p
    C_maxs = gubs.get_cmax_reachable(
        s0, V_risk, V_i, P_risk, pi_risk, goal, A, C, lamb, k_g, succ_states)
    c_max_values = [x for x in C_maxs.values()]
    max_c = max(c_max_values)
    mean_c = np.mean(c_max_values)
    # C_maxs = {k: max_c for k in C_maxs}
    # print("C_maxs:", C_maxs)

    # "unexpand" all states in mdp_obj
    #for s in mdp_obj:
    #    mdp_obj[s]['expanded'] = False

    solved = bool(C_maxs[s0] == 0)
    #if solved:
    #    mdp_obj[s0]['solved'] = True
    value = V_risk[V_i[s0]] if solved else 1
    prob = P_risk[V_i[s0]]
    pi = pi_risk[V_i[s0]] if solved else None

    bpsg = {(s0, 0): {"Adj": []}}
    explicit_graph = {
        (s0, 0): {
            "value": value,
            "prob": prob,
            "solved": solved,
            "expanded": False,
            "pi": pi,
            "Adj": []
        }
    }

    i = 0
    unexpanded = get_unexpanded_states_extended(
        goal, explicit_graph, bpsg)

    n_updates = 0
    old_n_updates = 0
    while len(unexpanded) > 0:
        print("i =", i)
        #print("Unexpanded states:", unexpanded)

        s = unexpanded[0]
        print("Will expand", text_render(env, s[0]), " with cost", s[1])
        print()
        # input()
        # print("Explicit graph before:", explicit_graph)
        #for s in unexpanded:
        #explicit_graph, C_maxs = expand_state_gubs_v2(
        #    s, h_v, env, goal, explicit_graph, C, C_maxs, V_risk, P_risk, pi_risk, V_i, A, k_g, lamb)
        for s in unexpanded:
            explicit_graph, C_maxs = expand_state_gubs_v2(
            s, h_v, env, goal, explicit_graph, C, C_maxs, V_risk, P_risk, pi_risk, V_i, A, k_g, lamb)
        # print("Explicit graph after:", explicit_graph)
        # print()
        # print("Best partial solution graph before:", bpsg)
        # print()
        #Z = [s]
        sorted_bpsg = list(reversed(topological_sort(bpsg)))
        unexpanded_set = set(unexpanded)
        Z = list(set([s for s in sorted_bpsg if s in unexpanded_set]))

        print("Z size =", len(Z))
        explicit_graph, n_updates_, _ = value_iteration_gubs(
            explicit_graph, V_i, A, Z, k_g, lamb, C, env)
        old_n_updates += len(Z)
        n_updates += n_updates_
        # print("Explicit graph after value iteration:", explicit_graph)
        # print()

        bpsg = update_partial_solution_extended(
            s0, C, bpsg, explicit_graph)
        # print("Best partial solution graph after:", bpsg)
        # print()
        unexpanded = get_unexpanded_states_extended(
            goal, explicit_graph, bpsg)

        i += 1

    return explicit_graph, bpsg, explicit_graph_dc, C_maxs, n_updates, n_updates_dc, old_n_updates
