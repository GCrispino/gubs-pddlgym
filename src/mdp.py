from copy import copy
from collections import deque
import gubs
import utils
import numpy as np
from pddlgym.core import get_successor_states, InvalidAction
from pddlgym.inference import check_goal
from numba import jit


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

    reach[s.literals] = {}
    for a in A:
        succ = get_successor_states_check_exception(s, a, env.domain)

        reach[s.literals][a] = {s_.literals: prob for s_, prob in succ.items()}
        for s_ in succ:
            if s_.literals not in reach:
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


def expand_state_dual_criterion(s,
                                h_v,
                                h_p,
                                env,
                                explicit_graph,
                                goal,
                                A,
                                p_zero=True,
                                succs_cache=None):
    if check_goal(utils.from_literals(s), goal):
        raise ValueError(
            f'State {s} can\'t be expanded because it is a goal state')

    succs_cache = {} if succs_cache == None else succs_cache
    neighbour_states_dict = {}
    neighbour_states = []
    i = 0
    for a in A:
        if succs_cache and (s, a) in succs_cache:
            succs = succs_cache[(s, a)]
        else:
            succs = get_successor_states_check_exception(
                utils.from_literals(s), a, env.domain)
        for s_, p in succs.items():
            if s_.literals not in neighbour_states_dict:
                neighbour_states_dict[s_.literals] = i
                i += 1
                neighbour_states.append({'state': s_.literals, 'A': {a: p}})
            else:
                neighbour_states[neighbour_states_dict[
                    s_.literals]]['A'][a] = p

    unexpanded_neighbours = filter(
        lambda _s: (not _s['state'] in explicit_graph) or
        (not explicit_graph[_s['state']]['expanded']), neighbour_states)

    # Add new empty states to 's' adjacency list
    new_explicit_graph = copy(explicit_graph)

    if p_zero:
        new_explicit_graph[s]['prob'] = 0

    new_explicit_graph[s]["Adj"].extend(neighbour_states)

    for n in unexpanded_neighbours:
        if n['state'] != s and n['state'] not in new_explicit_graph:
            is_goal = check_goal(utils.from_literals(n['state']), goal)
            h_v_ = 1 if is_goal else h_v(n['state'])
            h_p_ = 1 if is_goal else h_p(n['state'])
            new_explicit_graph[n['state']] = {
                "value": h_v_,
                "prob": h_p_,
                "solved": False,
                "pi": None,
                "expanded": False,
                "Q_v": {a: h_v_
                        for a in A},
                "Q_p": {a: h_p_
                        for a in A},
                "Adj": []
            }

    new_explicit_graph[s]['expanded'] = True

    return new_explicit_graph


def expand_state_gubs(s, env, goal, explicit_graph, C, C_maxs, V_risk, P_risk,
                      pi_risk, V_i, A):
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

    unexpanded_neighbours = list(
        filter(
            lambda _s: ((_s['state'] not in explicit_graph) or
                        (not explicit_graph[_s['state']]['expanded'])),
            neighbour_states))

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


def expand_state_gubs_v2(s,
                         h,
                         env,
                         goal,
                         explicit_graph,
                         C,
                         C_maxs,
                         V_risk,
                         P_risk,
                         pi_risk,
                         V_i,
                         A,
                         k_g,
                         lamb,
                         succs_cache=None,
                         approx=False):
    s_literals = utils.from_literals(s[0])
    if check_goal(s_literals, goal):
        raise ValueError(
            f'State {s[0]} can\'t be expanded because it is a goal state')

    succs_cache = {} if succs_cache == None else succs_cache
    # Get 's' neighbour states that were not expanded
    #    and "collapse" equal (s, C) pairs
    neighbour_states_dict = {}
    neighbour_states = []

    i = 0
    for a in A:
        if succs_cache and (s[0], a) in succs_cache:
            succs = succs_cache[(s[0], a)]
        else:
            succs = get_successor_states_check_exception(
                s_literals, a, env.domain)
            succs_cache[(s[0], a)] = succs
        for s_, p in succs.items():
            c_ = s[1] + C(s_.literals, a)
            if (s_.literals, c_) not in neighbour_states_dict:
                neighbour_states_dict[(s_.literals, c_)] = i
                i += 1
                neighbour_states.append({
                    'state': (s_.literals, c_),
                    'A': {
                        a: p
                    }
                })
            else:
                neighbour_states[neighbour_states_dict[(s_.literals,
                                                        c_)]]['A'][a] = p

    unexpanded_neighbours = list(
        filter(
            lambda _s: ((_s['state'] not in explicit_graph) or
                        (not explicit_graph[_s['state']]['expanded'])),
            neighbour_states))
    expanded_neighbours = list(
        filter(
            lambda _s: ((_s['state'] in explicit_graph) and explicit_graph[_s[
                'state']]['expanded']), neighbour_states))

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
        if callable(C_maxs):
            c_max_n = C_maxs(n[0])
        elif n[0] not in C_maxs:
            raise NotImplementedError()
            # Compute succ states
            #C_maxs = gubs.get_cmax_reachable(
            #    n[0], V_risk, V_i, P_risk, pi_risk, goal, A, C, lamb, k_g, succ_states, W_s=C_maxs)
        else:
            c_max_n = C_maxs[n[0]]

        is_goal = check_goal(utils.from_literals(n[0]), goal)
        solved = bool(is_goal or n[1] >= c_max_n)
        if approx:
            value = 1 if is_goal else V_risk(n[0])
            prob = 1 if is_goal else P_risk(n[0])
            pi = A[0] if is_goal else pi_risk(n[0])

        value = (V_risk(n[0]) if callable(V_risk) else
                 V_risk[V_i[n[0]]]) if solved else h(n[0])
        prob = P_risk(n[0]) if callable(P_risk) else P_risk[V_i[n[0]]]
        # TODO -> Melhorar esse A[0] aqui
        #           - Fazer ser opcional? Por ex só fazer isso quando algum parâmetro multiple_leves for verdadeiro, ou algo assim
        #           - Além disso, ao invés de escolher a primeira ação, escolher a melhor ação segundo a heurística
        pi = (pi_risk(n[0])
              if callable(pi_risk) else pi_risk[n[0]]) if solved else A[0]

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

    # TODO -> Construir unexpanded_set lá em cima, quando tiver construindo o unexpanded_neighbours
    unexpanded_set = set([
        n['state'] for n in unexpanded_neighbours
        if not check_goal(utils.from_literals(n['state'][0]), goal)
    ])

    return new_explicit_graph, C_maxs, succs_cache, unexpanded_set


def get_unexpanded_states(goal, explicit_graph, bpsg):
    return list(
        filter(
            lambda x: (x not in explicit_graph) or
            (not explicit_graph[x]["expanded"] and not check_goal(
                utils.from_literals(x) if type(x) == frozenset else x, goal)),
            bpsg.keys()))


def get_unexpanded_states_extended(goal, explicit_graph, bpsg):
    return list(
        filter(
            lambda x: x in explicit_graph and not explicit_graph[x]['expanded']
            and not explicit_graph[x]["solved"] and not check_goal(
                utils.from_literals(x[0]), goal), bpsg.keys()))


def find_reachable(s, a, mdp):
    """ Find states that are reachable from state 's' after executing action 'a' """
    all_reachable_from_s = mdp[s]['Adj']
    return list(filter(lambda obj_s_: a in obj_s_['A'], all_reachable_from_s))


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
    return list(
        filter(
            lambda s_: s_ != s and len(
                list(
                    filter(
                        lambda s__: s__['state'] == s and
                        (True if not best else graph[s_]['pi'] in s__['A']
                         ), graph[s_]['Adj']))) > 0,
            graph[s]['parents'] if 'parents' in graph[s] else graph))


def __find_ancestors_extended(s, bpsg, visited, C, best=False, sorting=None):
    sorting = [] if sorting == None else sorting
    direct_ancestors = list(
        filter(lambda a: a not in visited,
               find_direct_ancestors_extended(s, bpsg, best)))

    result = [] + direct_ancestors

    for a in direct_ancestors:
        if a not in visited:
            visited.add(a)
            result_, _ = __find_ancestors_extended(a, bpsg, visited, C, best,
                                                   sorting)
            result += result_
    sorting.append(s)

    return result, sorting


def find_ancestors_extended(s, bpsg, C, best=False):
    return __find_ancestors_extended(s, bpsg, set(), C, best)


def find_neighbours(s, adjs):
    """ Find neighbours of s in adjacency list (except itself) """
    return list(
        map(lambda s_: s_['state'], filter(lambda s_: s_['state'] != s, adjs)))


def find_unreachable(s0, mdp):
    """ search for unreachable states using dfs """
    S = list(mdp.keys())
    len_s = len(S)
    V_i = {S[i]: i for i in range(len_s)}
    colors = ['w'] * len_s
    dfs_visit(V_i[s0], colors, [-1] * len_s, [-1] * len_s, [-1] * len_s, [0],
              S, V_i, mdp)
    return [S[i] for i, c in enumerate(colors) if c != 'b']


def dfs_visit(i,
              colors,
              d,
              f,
              low,
              time,
              S,
              V_i,
              mdp,
              on_visit=None,
              on_visit_neighbor=None,
              on_finish=None):
    colors[i] = 'g'
    time[0] += 1
    d[i] = time[0]
    low[i] = time[0]
    s = S[i]

    if on_visit:
        on_visit(s, i, d, low)

    for s_obj in mdp[s]['Adj']:
        s_ = s_obj['state']
        if s_ not in mdp:
            continue
        j = V_i[s_]
        if colors[j] == 'w':
            dfs_visit(j, colors, d, f, low, time, S, V_i, mdp, on_visit,
                      on_visit_neighbor, on_finish)
            low[i] = min(low[i], low[j])
        if on_visit_neighbor:
            on_visit_neighbor(s, i, s_, j, d, low)

    if on_finish:
        on_finish(s, i, d, low)

    colors[i] = 'b'
    time[0] += 1
    f[i] = time[0]


def dfs(mdp, on_visit=None, on_visit_neighbor=None, on_finish=None):
    S = list(mdp.keys())
    len_s = len(S)
    V_i = {S[i]: i for i in range(len_s)}
    # (w)hite, (g)ray or (b)lack
    colors = ['w'] * len_s
    d = [-1] * len_s
    f = [-1] * len_s
    low = [-1] * len_s
    time = [0]
    for i in range(len_s):
        c = colors[i]
        if c == 'w':
            dfs_visit(i, colors, d, f, low, time, S, V_i, mdp, on_visit,
                      on_visit_neighbor, on_finish)

    return d, f, colors


def get_sccs(mdp):
    stack = deque()
    sccs = set()

    def on_visit(s, i, d, low):
        stack.append(s)

    def on_visit_neighbor(s, i, s_, j, d, low):
        if s_ in stack:
            low[i] = min(low[i], d[j])

    def on_finish(s, i, d, low):
        new_scc = deque()
        if d[i] == low[i]:
            while True:
                n = stack.pop()
                new_scc.append(n)
                if s == n:
                    break
        if len(new_scc) > 0:
            sccs.add(frozenset(new_scc))

    dfs(mdp, on_visit, on_visit_neighbor, on_finish)

    return sccs


def topological_sort(mdp):
    stack = []

    def dfs_fn(s, *_):
        stack.append(s)

    dfs(mdp, on_finish=dfs_fn)
    return list(reversed(stack))


def update_action_partial_solution(s, s0, bpsg, explicit_graph):
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

    return bpsg_


def update_partial_solution(s0, bpsg, explicit_graph):
    bpsg_ = copy(bpsg)

    for s in bpsg:
        a = explicit_graph[s]['pi']
        if s not in bpsg_:
            continue

        s_obj = bpsg_[s]

        if len(s_obj['Adj']) == 0:
            if a is not None:
                bpsg_ = update_action_partial_solution(s, s0, bpsg_,
                                                       explicit_graph)
        else:
            best_current_action = next(iter(s_obj['Adj'][0]['A'].keys()))

            if a is not None and best_current_action != a:
                bpsg_ = update_action_partial_solution(s, s0, bpsg_,
                                                       explicit_graph)

    unreachable = find_unreachable(s0, bpsg_)

    for s_ in unreachable:
        if s_ in bpsg_:
            bpsg_.pop(s_)

    return bpsg_


def update_action_partial_solution_extended(s, s0, C, bpsg, explicit_graph):
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
        reachable = find_reachable(s, a, explicit_graph)

        for s_obj_ in reachable:
            s_ = s_obj_['state']
            s_obj['Adj'].append({'state': s_, 'A': {a: s_obj_['A'][a]}})

            #if s_extended not in bpsg_:
            if s_ not in bpsg_:

                #bpsg_[s_extended] = {
                bpsg_[s_] = {"Adj": []}
                bpsg_[s] = s_obj

                if s_ in explicit_graph and explicit_graph[s_]['expanded']:
                    states.append(s_)
        i += 1

    return bpsg_


def update_partial_solution_extended(s0, C, bpsg, explicit_graph):
    bpsg_ = copy(bpsg)

    for s in bpsg:
        a = explicit_graph[s]['pi']
        if s not in bpsg_:
            continue

        s_obj = bpsg_[s]

        if len(s_obj['Adj']) == 0:
            if a is not None:
                bpsg_ = update_action_partial_solution_extended(
                    s, s0, C, bpsg_, explicit_graph)
        else:
            best_current_action = next(iter(s_obj['Adj'][0]['A'].keys()))

            if a is not None and best_current_action != a:
                bpsg_ = update_action_partial_solution_extended(
                    s, s0, C, bpsg_, explicit_graph)

    unreachable = find_unreachable((s0, 0), bpsg_)

    for s_ in unreachable:
        if s_ in bpsg_:
            bpsg_.pop(s_)

    return bpsg_


def is_trap(scc, sccs, goal, explicit_graph):
    """
        detect if there is an edge that goes to a state
        from other components that is not a goal
    """

    is_trap = True
    for s in scc:
        if check_goal(utils.from_literals(s), goal):
            is_trap = False
        for adj in explicit_graph[s]['Adj']:
            s_ = adj['state']
            if explicit_graph[s_]['scc'] != explicit_graph[s]['scc']:
                is_trap = False
                break
        if not is_trap:
            break
    return is_trap


def eliminate_traps(bpsg, goal, A, explicit_graph, env, succs_cache):
    sccs = list(get_sccs(bpsg))

    # store scc index for each state in bpsg
    for i, scc in enumerate(sccs):
        for s in scc:
            bpsg[s]['scc'] = i

    traps = set(filter(lambda scc: is_trap(scc, sccs, goal, bpsg), sccs))

    for i, trap in enumerate(traps):
        # check if trap is transient or permanent
        found_action = False
        actions = set()
        actions_diff_state = set()
        for s in trap:
            if not explicit_graph[s]['expanded']:
                all_succs = set()
                for a in A:
                    succs = get_successor_states_check_exception(
                        utils.from_literals(s), a, env.domain)
                    if (s, a) not in succs_cache:
                        succs_cache[(s, a)] = succs
                    all_succs.update(set(succs))
                    for s_ in succs:
                        if s_.literals != s:
                            actions_diff_state.add(a)
                        if s_.literals not in trap:
                            found_action = True
                            actions.add(a)
            else:
                for adj in explicit_graph[s]['Adj']:
                    s_ = adj['state']
                    if s_ != s:
                        actions_diff_state.update(set(adj['A']))
                    if s_ not in trap:
                        found_action = True
                        actions.update(set(adj['A']))

        if not found_action:
            # permanent
            for s in trap:
                if explicit_graph[s]['solved']:
                    continue
                explicit_graph[s]['value'] = 0
                explicit_graph[s]['prob'] = 0
                explicit_graph[s]['solved'] = True
        else:
            # transient
            shape = len(trap), len(actions)
            Q_v = np.zeros(shape)
            Q_p = np.zeros(shape)
            trap_states = list(trap)
            action_list = list(actions)
            for i, s in enumerate(trap_states):
                for i_a, a in enumerate(action_list):
                    Q_v[i][i_a] = explicit_graph[s]['Q_v'][a]
                    Q_p[i][i_a] = explicit_graph[s]['Q_p'][a]
            max_prob = np.max(Q_p)
            i_max_prob = np.argwhere(Q_p == max_prob)
            i_s_a_max_prob = {i: set() for i in set(i_max_prob.T[0])}
            for i_s, i_a in i_max_prob:
                i_s_a_max_prob[i_s].add(i_a)

            max_utility = -np.inf
            a_max_utility = None
            s_max_utility = None
            for i_s, i_a_set in i_s_a_max_prob.items():
                for i_a in i_a_set:
                    if Q_v[i_s, i_a] > max_utility:
                        max_utility = Q_v[i_s, i_a]
                        a_max_utility = action_list[i_a]
                        s_max_utility = trap_states[i_s]

            for s in trap:
                # if state is solved, skip it
                if explicit_graph[s]['solved']:
                    continue
                if 'blacklist' not in explicit_graph[s]:
                    explicit_graph[s]['blacklist'] = set()
                blacklist = explicit_graph[s]['blacklist']
                new_blacklisted = None
                if s == s_max_utility:
                    explicit_graph[s]['pi'] = a_max_utility

                    # blacklist actions that are not in actions set
                    new_blacklisted = set(A) - actions
                else:
                    # blacklist actions that go to the same state
                    new_blacklisted = set(A) - actions_diff_state
                assert len(new_blacklisted) < len(A)
                # mark as changed if blacklist set changes
                if new_blacklisted and (new_blacklisted != blacklist):
                    explicit_graph[s]['blacklist'] = new_blacklisted

                explicit_graph[s]['value'] = max_utility
                explicit_graph[s]['prob'] = max_prob

    return bpsg, succs_cache


def backup_prob(explicit_graph, A, s, goal, lamb, C):
    np.seterr(all='raise')

    A_blacklist = explicit_graph[s][
        'blacklist'] if 'blacklist' in explicit_graph[s] else set()
    all_reachable = np.array([find_reachable(s, a, explicit_graph) for a in A],
                             dtype=object)
    actions_results_p = np.array([
        sum_(
            np.array([
                compute_prob(explicit_graph[s_['state']]['prob'], s_['A'][a])
                for s_ in all_reachable[i]
            ])) for i, a in enumerate(A)
    ])

    # set maxprob
    max_prob = np.max([
        res for i, res in enumerate(actions_results_p)
        if A[i] not in A_blacklist
    ])
    explicit_graph[s]['prob'] = max_prob
    i_A_max_prob = np.array([
        i for i, res in enumerate(actions_results_p)
        if res == max_prob and A[i] not in A_blacklist
    ])

    A_max_prob = A[i_A_max_prob]

    for i, a in enumerate(A):
        explicit_graph[s]['Q_p'][a] = actions_results_p[i]

    explicit_graph[s]['pi'] = A_max_prob[0]

    return explicit_graph


@jit(nopython=True)
def sum_(a):
    return np.sum(a)


@jit(nopython=True)
def compute_state_value(v, p, c, lamb):
    return np.exp(lamb * c) * v * p


@jit(nopython=True)
def compute_prob(p, p_):
    return p * p_


def value_iteration_dual_criterion(explicit_graph,
                                   bpsg,
                                   A,
                                   Z,
                                   goal,
                                   lamb,
                                   C,
                                   epsilon=1e-3,
                                   p_zero=True,
                                   n_iter=None,
                                   convergence_test=False):
    n_states = len(explicit_graph)
    n_actions = len(A)

    # initialize
    V = np.zeros(n_states, dtype=float)
    Q_v = np.zeros((n_states, n_actions), dtype=float)
    Q_p = np.zeros((n_states, n_actions), dtype=float)
    pi = np.full(n_states, None)
    P = np.zeros(n_states, dtype=float)
    V_i = {s: i for i, s in enumerate(explicit_graph)}
    A_i = {a: i for i, a in enumerate(A)}

    if p_zero:
        for s in Z:
            if not check_goal(utils.from_literals(s),
                              goal) and not explicit_graph[s][
                                  'solved'] and explicit_graph[s]['expanded']:
                explicit_graph[s]['prob'] = 0
                for a in A:
                    explicit_graph[s]['Q_p'][a] = 0

    for s, n in explicit_graph.items():
        V[V_i[s]] = n['value']
        P[V_i[s]] = n['prob']
        pi[V_i[s]] = n['pi']

        for a in A:
            Q_v[V_i[s], A_i[a]] = n['Q_v'][a]
            Q_p[V_i[s], A_i[a]] = n['Q_p'][a]

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
            all_reachable = np.array(
                [find_reachable(s, a, explicit_graph) for a in A],
                dtype=object)
            if not p_zero and explicit_graph[s]['pi']:
                solved_succ = [
                    explicit_graph[s_['state']]['solved']
                    for s_ in all_reachable[A_i[explicit_graph[s]['pi']]]
                ]
                if all(solved_succ):
                    explicit_graph[s]['solved'] = True
                    continue

            A_blacklist = explicit_graph[s][
                'blacklist'] if 'blacklist' in explicit_graph[s] else set()
            n_updates += 1
            actions_results_p = np.array([
                sum_(
                    np.array([
                        compute_prob(P[V_i[s_['state']]], s_['A'][a])
                        for s_ in all_reachable[i]
                    ])) for i, a in enumerate(A)
            ])
            Q_p[V_i[s]] = actions_results_p

            # set maxprob
            max_prob = np.max([
                res for i, res in enumerate(actions_results_p)
                if A[i] not in A_blacklist
            ])
            P_[V_i[s]] = max_prob
            i_A_max_prob = np.array([
                i for i, res in enumerate(actions_results_p)
                if res == max_prob and A[i] not in A_blacklist
            ])

            A_max_prob = A[i_A_max_prob]
            not_max_prob_actions_results = np.array([
                res for i, res in enumerate(actions_results_p)
                if A[i] not in A_blacklist
            ])

            P_not_max_prob[V_i[s]] = P[V_i[s]] if len(
                not_max_prob_actions_results) == 0 else np.max(
                    not_max_prob_actions_results)

            actions_results = np.array([
                sum_(
                    np.array([
                        compute_state_value(V[V_i[s_['state']]], s_['A'][a],
                                            C(s, A[i]), lamb)
                        for s_ in all_reachable[i]
                        #np.exp(lamb * C(s, A[i])) * V[V_i[s_['state']]] *
                        #s_['A'][a] for s_ in all_reachable[i]
                        #]) for i in i_A_max_prob
                    ])) for i, a in enumerate(A)
            ])
            actions_results_max_prob = actions_results[i_A_max_prob]
            Q_v[V_i[s]] = actions_results

            i_a = np.argmax(actions_results_max_prob)
            V_[V_i[s]] = actions_results_max_prob[i_a]
            pi_[V_i[s]] = A_max_prob[i_a]

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

        for a in A:
            explicit_graph[s]['Q_v'][a] = Q_v[V_i[s], A_i[a]]
            explicit_graph[s]['Q_p'][a] = Q_p[V_i[s], A_i[a]]

    #print(f'{i} iterations')
    return explicit_graph, converged, changed, n_updates


def lao_dual_criterion_fret(s0,
                            h_v,
                            h_p,
                            goal,
                            A,
                            lamb,
                            env,
                            epsilon=1e-3,
                            explicit_graph=None,
                            succs_cache=None):
    bpsg = {s0: {"Adj": []}}
    explicit_graph = explicit_graph or {}
    succs_cache = {} if succs_cache == None else succs_cache

    if s0 in explicit_graph and explicit_graph[s0]['solved']:
        return explicit_graph, bpsg, 0, succs_cache

    if s0 not in explicit_graph:
        explicit_graph[s0] = {
            "value": h_v(s0),
            "prob": h_p(s0),
            "solved": False,
            "expanded": False,
            "pi": None,
            "Q_v": {a: h_v(s0)
                    for a in A},
            "Q_p": {a: h_p(s0)
                    for a in A},
            "Adj": []
        }

    def C(s, a):
        s = utils.from_literals(s) if type(s) == frozenset else s
        return 0 if check_goal(s, goal) else 1

    i = 1
    unexpanded = get_unexpanded_states(goal, explicit_graph, bpsg)
    n_updates = 0
    explicit_graph_cur_size = 1
    while True:
        while len(unexpanded) > 0:
            s = unexpanded[0]
            print("Iteration", i)
            print("Will expand", len(unexpanded), "states")
            Z = set()
            for s in unexpanded:
                explicit_graph = expand_state_dual_criterion(
                    s,
                    h_v,
                    h_p,
                    env,
                    explicit_graph,
                    goal,
                    A,
                    p_zero=False,
                    succs_cache=succs_cache)
                Z.add(s)
                Z.update(find_ancestors(s, explicit_graph, best=True))

            assert len(explicit_graph) >= explicit_graph_cur_size

            explicit_graph_cur_size = len(explicit_graph)
            print("explicit graph size:", explicit_graph_cur_size)
            print("Z size:", len(Z))
            explicit_graph, _, __, n_updates_ = value_iteration_dual_criterion(
                explicit_graph,
                bpsg,
                A,
                Z,
                goal,
                lamb,
                C,
                epsilon=epsilon,
                p_zero=False)
            print(f"Finished value iteration in {n_updates_} updates")
            n_updates += n_updates_
            bpsg = update_partial_solution(s0, bpsg, explicit_graph)

            bpsg, succs_cache = eliminate_traps(bpsg, goal, A, explicit_graph,
                                                env, succs_cache)

            bpsg = update_partial_solution(s0, bpsg, explicit_graph)

            unexpanded = get_unexpanded_states(goal, explicit_graph, bpsg)
            i += 1
        bpsg_states = [
            s_ for s_ in bpsg.keys()
            if not check_goal(utils.from_literals(s_), goal)
        ]
        print(f"Will start convergence test for bpsg with {len(bpsg)} states")
        explicit_graph, converged, changed, n_updates_ = value_iteration_dual_criterion(
            explicit_graph,
            bpsg,
            A,
            bpsg_states,
            goal,
            lamb,
            C,
            epsilon=epsilon,
            convergence_test=True,
            p_zero=False)
        n_updates += n_updates_

        bpsg = update_partial_solution(s0, bpsg, explicit_graph)
        bpsg, succs_cache = eliminate_traps(bpsg, goal, A, explicit_graph, env,
                                            succs_cache)

        bpsg = update_partial_solution(s0, bpsg, explicit_graph)
        unexpanded = get_unexpanded_states(goal, explicit_graph, bpsg)

        if changed:
            continue

        if converged and len(unexpanded) == 0:
            break
    for s_ in bpsg:
        explicit_graph[s_]['solved'] = True
    return explicit_graph, bpsg, n_updates, succs_cache


def lao_dual_criterion(s0,
                       h_v,
                       h_p,
                       goal,
                       A,
                       lamb,
                       env,
                       epsilon=1e-3,
                       p_zero=True,
                       explicit_graph=None,
                       succs_cache=None):
    bpsg = {s0: {"Adj": []}}
    explicit_graph = explicit_graph or {}
    succs_cache = {} if succs_cache == None else succs_cache

    if s0 in explicit_graph and explicit_graph[s0]['solved']:
        return explicit_graph, bpsg, 0, succs_cache

    if s0 not in explicit_graph:
        explicit_graph[s0] = {
            "value": h_v(s0),
            "prob": 0,
            "solved": False,
            "expanded": False,
            "pi": None,
            "Q_v": {a: h_v(s0)
                    for a in A},
            "Q_p": {a: h_p(s0)
                    for a in A},
            "Adj": []
        }

    def C(s, a):
        s = utils.from_literals(s) if type(s) == frozenset else s
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
            print("Will expand", len(unexpanded), "states")
            Z = set()
            for s in unexpanded:
                explicit_graph = expand_state_dual_criterion(
                    s,
                    h_v,
                    h_p,
                    env,
                    explicit_graph,
                    goal,
                    A,
                    succs_cache=succs_cache)
                Z.add(s)
                Z.update(find_ancestors(s, explicit_graph, best=True))
            #Z = [s] + find_ancestors(s, explicit_graph, best=True)
            assert len(explicit_graph) >= explicit_graph_cur_size
            explicit_graph_cur_size = len(explicit_graph)
            print("explicit graph size:", explicit_graph_cur_size)
            print("Z size:", len(Z))
            explicit_graph, _, __, n_updates_ = value_iteration_dual_criterion(
                explicit_graph,
                bpsg,
                A,
                Z,
                goal,
                lamb,
                C,
                epsilon=epsilon,
                p_zero=p_zero)
            print(f"Finished value iteration in {n_updates_} updates")
            n_updates += n_updates_
            bpsg = update_partial_solution(s0, bpsg, explicit_graph)
            unexpanded = get_unexpanded_states(goal, explicit_graph, bpsg)
            i += 1
        bpsg_states = [
            s_ for s_ in bpsg.keys()
            if not check_goal(utils.from_literals(s_), goal)
        ]
        print(f"Will start convergence test for bpsg with {len(bpsg)} states")
        explicit_graph, converged, changed, n_updates_ = value_iteration_dual_criterion(
            explicit_graph,
            bpsg,
            A,
            bpsg_states,
            goal,
            lamb,
            C,
            epsilon=epsilon,
            p_zero=p_zero,
            convergence_test=True)
        print(f"Finished convergence test in {n_updates_} updates")
        n_updates += n_updates_

        bpsg = update_partial_solution(s0, bpsg, explicit_graph)
        unexpanded = get_unexpanded_states(goal, explicit_graph, bpsg)

        if converged and len(unexpanded) == 0:
            break
    for s_ in bpsg:
        explicit_graph[s_]['solved'] = True
    return explicit_graph, bpsg, n_updates, succs_cache


def lao_dual_criterion_reachable(s0,
                                 h_v,
                                 h_p,
                                 goal,
                                 A,
                                 lamb,
                                 env,
                                 epsilon=1e-3,
                                 eliminate_traps=False,
                                 ilao=False):
    #all_reachable = mg.find_all_reachable(s0, mdp_obj)
    all_reachable = set({s0})
    explicit_graph = {}

    stack = [s0]
    n_updates_total = 0
    succs_cache = {}
    while len(stack) > 0:
        s = stack.pop()
        if s in explicit_graph and explicit_graph[s]['solved']:
            continue

        print("Will call lao_dual_criterion for state:",
              utils.text_render(env, utils.from_literals(s)))
        if ilao:
            explicit_graph, _, n_updates, succs_cache = ilao_dual_criterion_fret(
                s,
                h_v,
                h_p,
                goal,
                A,
                lamb,
                env,
                epsilon=epsilon,
                explicit_graph=explicit_graph,
                succs_cache=succs_cache)
        else:
            if eliminate_traps:
                explicit_graph, _, n_updates, succs_cache = lao_dual_criterion_fret(
                    s,
                    h_v,
                    h_p,
                    goal,
                    A,
                    lamb,
                    env,
                    epsilon=epsilon,
                    explicit_graph=explicit_graph)
            else:
                explicit_graph, _, n_updates, succs_cache = lao_dual_criterion(
                    s,
                    h_v,
                    h_p,
                    goal,
                    A,
                    lamb,
                    env,
                    epsilon=epsilon,
                    explicit_graph=explicit_graph)
        n_updates_total += n_updates
        print(' finished lao dual criterion for',
              utils.text_render(env, utils.from_literals(s)),
              explicit_graph[s]['prob'], explicit_graph[s]['value'],
              len([v for v in explicit_graph.values() if v['solved']]))
        for s_ in explicit_graph:
            if s_ not in all_reachable:
                all_reachable.add(s_)
                stack.append(s_)

    #pi = {s: explicit_graph[s]['pi']
    #      for s in sorted(explicit_graph, key=int)}
    #print(' pi:', pi)
    return explicit_graph, n_updates_total, succs_cache


def value_iteration_gubs(explicit_graph, A, Z, k_g, lamb, C, env):
    n_actions = len(A)
    changed = False
    stack = copy(Z)
    z_graph = {s: {'Adj': []} for s in stack}

    i = 0
    while len(stack) > 0:
        s = stack.pop(0)
        z_graph.pop(s)

        if explicit_graph[s]['solved']:
            print("SOLVED", utils.text_render(env, s[0]), s[1])
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
            gen_p = (s_['A'][a] * explicit_graph[s_['state']]['prob']
                     for s_ in reachable)
            p_actions_results[i_a] = np.sum(np.fromiter(gen_p, dtype=np.float))

        i_a_opt = np.argmax(
            np.exp(lamb * c) * q_actions_results + k_g * p_actions_results)

        reachable_opt = all_reachable[i_a_opt]
        c_s_a = C(s[0], A[i_a_opt])
        c_ = c + c_s_a
        #is_solved = all([explicit_graph[(r['name'], c_)]['solved']
        #                 for r in reachable_opt])
        is_solved = all(
            [explicit_graph[r['state']]['solved'] for r in reachable_opt])

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
            print(
                f"{utils.text_render(env, utils.from_literals(s[0]))} with cost {s[1]} is now solved!"
            )
        explicit_graph[s]['solved'] = is_solved

        if explicit_graph[s]['value'] < old_val or is_solved:
            # Label as solved and add ancestors to stack
            ancestors = [
                a for a in find_direct_ancestors_extended(
                    s, explicit_graph, best=True) if a not in stack
            ]
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


def ilao_dual_criterion_fret(s0,
                             h_v,
                             h_p,
                             goal,
                             A,
                             lamb,
                             env,
                             epsilon=1e-3,
                             explicit_graph=None,
                             succs_cache=None):
    bpsg = {s0: {"Adj": []}}
    explicit_graph = explicit_graph or {}
    succs_cache = {} if succs_cache == None else succs_cache

    if s0 in explicit_graph and explicit_graph[s0]['solved']:
        return explicit_graph, bpsg, 0, succs_cache

    if s0 not in explicit_graph:
        explicit_graph[s0] = {
            "value": h_v(s0),
            "prob": h_p(s0),
            "solved": False,
            "expanded": False,
            "pi": None,
            "Q_v": {a: h_v(s0)
                    for a in A},
            "Q_p": {a: h_p(s0)
                    for a in A},
            "Adj": []
        }

    def C(s, a):
        s = utils.from_literals(s) if type(s) == frozenset else s
        return 0 if check_goal(s, goal) else 1

    i = 1
    unexpanded = get_unexpanded_states(goal, explicit_graph, bpsg)
    n_updates = 0
    explicit_graph_cur_size = 1
    while True:
        while len(unexpanded) > 0:
            s = unexpanded[0]
            print("Iteration", i)
            print(len(unexpanded), "unexpanded states")

            n_updates_ = 0

            def visit(s, i, d, low):
                nonlocal explicit_graph, A, goal, n_updates_
                is_goal = check_goal(utils.from_literals(s), goal)
                if not is_goal and not explicit_graph[s]['expanded']:
                    explicit_graph = expand_state_dual_criterion(
                        s,
                        h_v,
                        h_p,
                        env,
                        explicit_graph,
                        goal,
                        A,
                        p_zero=False,
                        succs_cache=succs_cache)
                if not is_goal and not explicit_graph[s]['solved']:
                    # run bellman backup
                    explicit_graph = backup_prob(explicit_graph, A, s, goal,
                                                 lamb, C)
                    n_updates_ += 1

            dfs(bpsg, on_visit=visit)

            assert len(explicit_graph) >= explicit_graph_cur_size

            explicit_graph_cur_size = len(explicit_graph)
            print("explicit graph size:", explicit_graph_cur_size)
            print(f"Finished value iteration in {n_updates_} updates")
            n_updates += n_updates_
            bpsg = update_partial_solution(s0, bpsg, explicit_graph)

            bpsg, succs_cache = eliminate_traps(bpsg, goal, A, explicit_graph,
                                                env, succs_cache)

            bpsg = update_partial_solution(s0, bpsg, explicit_graph)

            unexpanded = get_unexpanded_states(goal, explicit_graph, bpsg)
            i += 1
        bpsg_states = [
            s_ for s_ in bpsg.keys()
            if not check_goal(utils.from_literals(s_), goal)
        ]
        print(f"Will start convergence test for bpsg with {len(bpsg)} states")
        explicit_graph, converged, changed, n_updates_ = value_iteration_dual_criterion(
            explicit_graph,
            bpsg,
            A,
            bpsg_states,
            goal,
            lamb,
            C,
            epsilon=epsilon,
            convergence_test=True,
            p_zero=False)
        n_updates += n_updates_
        print(f"Finished convergence test in {n_updates_} updates")

        bpsg = update_partial_solution(s0, bpsg, explicit_graph)
        bpsg, succs_cache = eliminate_traps(bpsg, goal, A, explicit_graph, env,
                                            succs_cache)

        bpsg = update_partial_solution(s0, bpsg, explicit_graph)
        unexpanded = get_unexpanded_states(goal, explicit_graph, bpsg)

        if changed:
            continue

        if converged and len(unexpanded) == 0:
            break
    for s_ in bpsg:
        explicit_graph[s_]['solved'] = True
    return explicit_graph, bpsg, n_updates, succs_cache


def egubs_ao(s0,
             h_v,
             h_p,
             goal,
             A,
             k_g,
             lamb,
             env,
             explicit_graph_dc,
             n_updates_dc,
             succs_cache,
             epsilon=1e-3,
             eliminate_traps=False,
             ilao_dc=False,
             expansion_levels=1):

    V_risk = {s: explicit_graph_dc[s]['value'] for s in explicit_graph_dc}
    P_risk = {s: explicit_graph_dc[s]['prob'] for s in explicit_graph_dc}
    pi_risk = {s: explicit_graph_dc[s]['pi'] for s in explicit_graph_dc}
    V_i = {s: s for s in V_risk}

    def C(s, a):
        s = utils.from_literals(s) if type(s) == frozenset else s
        return 0 if check_goal(s, goal) else 1

    succ_states = {}
    for s_ in explicit_graph_dc:
        for s__ in explicit_graph_dc[s_]['Adj']:
            for a, p in s__['A'].items():
                if (s_, a) not in succ_states:
                    succ_states[s_, a] = {}
                if s__['state'] not in succ_states[s_, a]:
                    succ_states[s_, a][s__['state']] = p
    C_maxs, W_s = gubs.get_cmax_reachable(s0, V_risk, V_i, P_risk,
                                          goal, A, C, lamb, k_g, succ_states)
    c_max_values = [x for x in C_maxs.values()]
    max_c = max(c_max_values)
    mean_c = np.mean(c_max_values)
    # C_maxs = {k: max_c for k in C_maxs}
    # print("C_maxs:", C_maxs)

    solved = bool(C_maxs[s0] == 0)
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
    unexpanded = set(get_unexpanded_states_extended(goal, explicit_graph,
                                                    bpsg))

    n_updates = 0
    old_n_updates = 0
    while not explicit_graph[(s0, 0)]['solved']:
        print("i =", i)

        if len(unexpanded) > 0:
            s = list(unexpanded)[0]
            print("Will expand",
                  utils.text_render(env, utils.from_literals(s[0])),
                  " with cost", s[1])
            print()

            # - Expansion:
            #       If expansion_levels > 1, more than one level in the state tree is expanded in the same iteration
            #       For each level, states are expanded and newly found states are added to a unexpanded set
            total_unexpanded = set()
            new_unexpanded_set = set()
            for level in range(expansion_levels):
                total_unexpanded.update(unexpanded)
                print(f"Level: {level},unexpanded: {len(total_unexpanded)}")
                while True:
                    try:
                        s = unexpanded.pop()
                    except KeyError:
                        break

                    explicit_graph, C_maxs, succs_cache, new_unexpanded = expand_state_gubs_v2(
                        s, h_v, env, goal, explicit_graph, C, C_maxs, V_risk,
                        P_risk, pi_risk, V_i, A, k_g, lamb, succs_cache)

                    new_unexpanded_set.update(new_unexpanded)

                unexpanded = new_unexpanded_set
                new_unexpanded_set = set()

            # Update BPSG with new expanded states
            bpsg = update_partial_solution_extended(s0, C, bpsg,
                                                    explicit_graph)

            sorted_bpsg = list(reversed(topological_sort(bpsg)))
            unexpanded_set = set(total_unexpanded)

            Z = list(set([s for s in sorted_bpsg if s in unexpanded_set]))
        else:
            sorted_bpsg = list(reversed(topological_sort(bpsg)))
            Z = sorted_bpsg

        print("Z size =", len(Z))
        explicit_graph, n_updates_, _ = value_iteration_gubs(
            explicit_graph, A, Z, k_g, lamb, C, env)
        old_n_updates += len(Z)
        n_updates += n_updates_

        bpsg = update_partial_solution_extended(s0, C, bpsg, explicit_graph)
        unexpanded = set(
            get_unexpanded_states_extended(goal, explicit_graph, bpsg))

        i += 1

    return explicit_graph, bpsg, explicit_graph_dc, C_maxs, n_updates, n_updates_dc, old_n_updates


def egubs_ao_approx(s0,
                    h_v,
                    h_p,
                    goal,
                    A,
                    k_g,
                    lamb,
                    env,
                    C_max,
                    epsilon=1e-3):
    def C_maxs(_):
        return C_max

    def V_risk(s):
        return h_v(s)

    def P_risk(s):
        return h_p(s)

    def pi_risk(_):
        return A[0]

    def C(s, a):
        s = utils.from_literals(s) if type(s) == frozenset else s
        return 0 if check_goal(s, goal) else 1

    bpsg = {(s0, 0): {"Adj": []}}
    explicit_graph = {
        (s0, 0): {
            "value": h_v(s0),
            "prob": h_p(s0),
            "solved": False,
            "expanded": False,
            "pi": None,
            "Adj": []
        }
    }

    i = 0
    unexpanded = get_unexpanded_states_extended(goal, explicit_graph, bpsg)

    n_updates = 0
    old_n_updates = 0
    succs_cache = {}
    while len(unexpanded) > 0:
        print("i =", i)

        s = unexpanded[0]
        print("Will expand", utils.text_render(env, utils.from_literals(s[0])),
              " with cost", s[1])
        print()
        for s in unexpanded:
            explicit_graph, C_maxs, succs_cache, _ = expand_state_gubs_v2(
                s, h_v, env, goal, explicit_graph, C, C_maxs, V_risk, P_risk,
                pi_risk, None, A, k_g, lamb, succs_cache, True)

        sorted_bpsg = list(reversed(topological_sort(bpsg)))
        unexpanded_set = set(unexpanded)
        Z = list(set([s for s in sorted_bpsg if s in unexpanded_set]))

        print("Z size =", len(Z))
        explicit_graph, n_updates_, _ = value_iteration_gubs(
            explicit_graph, A, Z, k_g, lamb, C, env)
        old_n_updates += len(Z)
        n_updates += n_updates_

        bpsg = update_partial_solution_extended(s0, C, bpsg, explicit_graph)

        unexpanded = get_unexpanded_states_extended(goal, explicit_graph, bpsg)

        i += 1

    return explicit_graph, bpsg, n_updates, old_n_updates


def build_explicit_graph_from_functions(V_dual,
                                        P_dual,
                                        pi_dual,
                                        V_i,
                                        S,
                                        A,
                                        env,
                                        goal,
                                        succs_cache,
                                        p_zero=True):
    def h_1(s):
        return 1

    for (s, a), val in succs_cache.items():
        succs_cache[(s, a)] = {
            utils.from_literals(s_): val_
            for s_, val_ in val.items()
        }
    explicit_graph = {}
    for s in S:
        if check_goal(utils.from_literals(s), goal):
            continue
        # Fill placeholder values just so expand_state_dual_criterion doesn't crash
        explicit_graph[s] = {
            "value": 0,
            "prob": 0,
            "solved": False,
            "expanded": False,
            "pi": None,
            "Adj": []
        }
        explicit_graph = expand_state_dual_criterion(s, h_1, h_1, env,
                                                     explicit_graph, goal, A,
                                                     p_zero, succs_cache)
        explicit_graph[s] = {
            **explicit_graph[s],
            "value": V_dual[V_i[s]],
            "prob": P_dual[V_i[s]],
            "solved": True,
            "expanded": True,
            "pi": pi_dual[V_i[s]],
            #"Q_v": {a: h_v(s) for a in A},
            #"Q_p": {a: h_p(s) for a in A},
        }

    return explicit_graph
