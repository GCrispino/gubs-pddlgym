import itertools
import math
from utils import get_values
import utils
import numpy as np
from pddlgym.inference import check_goal

def dual_criterion(lamb, V_i, S, h_v, goal, succ_states, A, c=1, epsilon=1e-3, n_iter=None):
    def u(c): return np.exp(lamb * c)

    G_i = [V_i[s] for s in V_i if check_goal(utils.from_literals(s), goal)]
    not_goal = [s for s in S if not check_goal(utils.from_literals(s), goal)]
    n_states = len(S)

    # initialize
    V = np.zeros(n_states, dtype=float)
    for s in not_goal:
        V[V_i[s]] = h_v(s)
    pi = np.full(n_states, None)
    P = np.zeros(n_states, dtype=float)
    V[G_i] = -np.sign(lamb)
    P[G_i] = 1
    if not isinstance(A, np.ndarray):
        A = np.array(A)

    i = 1

    P_not_max_prob = np.copy(P)
    while True:
        V_ = np.copy(V)
        P_ = np.copy(P)
        for s in not_goal:

            all_reachable = [succ_states[s, a] for a in A]
            actions_results_p = np.array([
                np.sum([
                    P[V_i[s_]] * p for s_, p in all_reachable[i].items()
                ]) for i, a in enumerate(A)
            ])

            # set maxprob
            max_prob = np.max(actions_results_p)
            P_[V_i[s]] = max_prob
            i_A_max_prob = np.argwhere(
                actions_results_p == max_prob).reshape(-1)
            A_max_prob = A[i_A_max_prob]
            not_max_prob_actions_results = actions_results_p[actions_results_p != max_prob]

            P_not_max_prob[V_i[s]] = P[V_i[s]] if len(not_max_prob_actions_results) == 0 else np.max(
                not_max_prob_actions_results)

            actions_results = np.array([
                np.sum([
                    u(c) * V[V_i[s_]] * p for s_, p in all_reachable[j].items()
                ]) for j in i_A_max_prob
            ])

            i_a = np.argmax(actions_results)
            V_[V_i[s]] = actions_results[i_a]
            pi[V_i[s]] = A_max_prob[i_a]

        v_norm = np.linalg.norm(V_ - V, np.inf)
        p_norm = np.linalg.norm(P_ - P, np.inf)

        P_diff = P_ - P_not_max_prob
        arg_min_p_diff = np.argmin(P_diff)
        min_p_diff = P_diff[arg_min_p_diff]

        if n_iter and i == n_iter:
            break
        print("Iteration", i)
        print(' delta1:', v_norm, p_norm, v_norm + p_norm)
        print(' delta2:', min_p_diff)
        #print('prob:', P, P_)
        if v_norm + p_norm < epsilon and min_p_diff >= 0:
            break
        V = V_
        P = P_
        i += 1

    #print(f'{i} iterations')
    return V, P, pi, i

def u(lamb, c): return np.exp(lamb * c)

def get_X(V, V_i, lamb, S, succ_states, A, c=1):

    list_X = [
        (
            (s, a),
            (V[V_i[s]] - np.sum(
                #np.fromiter(
                #    (s_['A'][a] * u(lamb, c) * V[V_i[s_['name']]]
                #     for s_ in mdp.find_reachable(s, a, mdp_obj)), dtype=float))
                np.fromiter(
                    (p * u(lamb, c) * V[V_i[s_]]
                     for s_, p in succ_states[s, a].items()), dtype=float))
             )
        )
        for (s, a) in itertools.product(S, A)
    ]

    X = np.array(list_X, dtype=object)

    return X[X.T[1] < 0]

def get_P_diff_W(s, a, P1, P2, V_i, k_g, succ_s):
    return k_g * (np.sum(np.fromiter((p * P1[V_i[s_]] for s_, p in succ_s.items()), dtype=float)) - P2[V_i[s]])

def get_cmax(V, V_i, P, S, succ_states, A, lamb, k_g, c=1):
    X = get_X(V, V_i, lamb, S, succ_states, A)
    W = np.full(len(X), -math.inf)

    for i, ((s, a), x) in enumerate(X):
        denominator = get_P_diff_W(s, a, P, P, V_i, k_g, succ_states[s, a])
        if denominator == 0:
            W[i] = -math.inf
        else:
            W[i] = -(1 / lamb) * np.log(
                x / denominator
            )

    try:
        C_max = np.max(W[np.invert(np.isnan(W))])
    except:
        return -math.inf
    if C_max == np.inf:
        raise Exception("this shouldn't happen")

    return int(np.ceil(C_max)) if C_max != -math.inf else -math.inf

def get_cmax_all(V, V_i, P, S, A, lamb, k_g, succ_states, c=1):

    X = get_X(V, V_i, lamb, S, succ_states, A)
    #print("X:", X)
    W = {}

    for (s, a), x in X:
        #denominator = k_g * (np.sum(np.fromiter((s_['A'][a] * P[V_i[s_['name']]]
        #                                         for s_ in mg.find_reachable(s, a, mdp_obj)), dtype=float)) - P[V_i[s]])
        denominator = k_g * (np.sum(np.fromiter((p * P[V_i[s_]]
                                                 for s_, p in succ_states[s, a].items()), dtype=float)) - P[V_i[s]])
        if denominator == 0:
            if s not in W:
                W[s] = {a: 0}
            W[s][a] = 0
        else:
            w = -(1 / lamb) * np.log(
                x / denominator
            )
            if not np.isnan(w):
                val = max(0, np.ceil(w).astype(int))
                if s not in W:
                    W[s] = {a: val}
                W[s][a] = val

    return W



def egubs_vi(V_dual, P_dual, pi_dual, C_max, lamb, k_g, V_i, S, goal, succ_states, A, c=1):
    G_i = [V_i[s] for s in V_i if check_goal(utils.from_literals(s), goal)]
    n_states = len(S)
    n_actions = len(A)

    C_max_plus = max(C_max, 0)

    V = np.zeros((n_states, C_max_plus + 1))
    V_dual_C = np.zeros((n_states, C_max_plus + 2))
    P = np.zeros((n_states, C_max_plus + 2))
    pi = np.full((n_states, C_max_plus + 2), None)

    #print(C_max, len(G_i), V_dual_C.shape)
    for i in range(V_dual_C.shape[1]):
        V_dual_C[G_i, i] = V_dual[G_i]

    #V_dual_C[G_i, :] = V_dual[G_i]
    P[G_i, :] = 1
    V_dual_C[:, C_max_plus + 1] = V_dual.T
    P[:, C_max_plus + 1] = P_dual.T
    pi[:, C_max_plus + 1] = pi_dual.T

    n_updates = 0
    for C in reversed(range(C_max_plus + 1)):
        Q = np.zeros(n_actions)
        P_a = np.zeros(n_actions)
        for s in S:
            i_s = V_i[s]
            n_updates += 1
            for i_a, a in enumerate(A):
                #c__ = 0 if mdp_obj[s]['goal'] else c
                c__ = 0 if check_goal(utils.from_literals(s), goal) else c
                c_ = C + c__
                successors = succ_states[s, a]

                # Get value
                gen_q = [p * V_dual_C[V_i[s_], c_]
                         for s_, p in successors.items()]
                #print(' gen_q:', gen_q, lamb, c__)
                Q[i_a] = u(lamb, c__) * \
                    np.sum(np.fromiter(gen_q, dtype=np.float))

                # Get probability
                gen_p = (p * P[V_i[s_], c_]
                         for s_, p in successors.items())
                P_a[i_a] = np.sum(
                    np.fromiter(gen_p, dtype=np.float)
                )
            i_a_opt = np.argmax(u(lamb, C) * Q + k_g * P_a)
            a_opt = A[i_a_opt]
            pi[i_s, C] = a_opt

            P[i_s, C] = P_a[i_a_opt]
            V_dual_C[i_s, C] = Q[i_a_opt]
            V[i_s, C] = V_dual_C[i_s, C] + k_g * P[i_s, C]

    return V, P, pi

def W(s, a, V_diff, V_i, P, k_g, lamb, succ_states):
    #denominator = k_g * (np.sum(np.fromiter((s_['A'][a] * P[V_i[s_['name']]]
    #                                         for s_ in mg.find_reachable(s, a, mdp_obj)), dtype=float)) - P[V_i[s]])
    denominator = k_g * (np.sum(np.fromiter((p * P[V_i[s_]]
                                             for s_, p in succ_states[s, a].items()), dtype=float)) - P[V_i[s]])

    W = None
    if denominator == 0:
        W = -math.inf
    else:
        W = np.nan
        try:
            W = -(1 / lamb) * np.log(
                V_diff / denominator
            )
        except FloatingPointError:
            pass
        if not np.isnan(W):
            W = max(-math.inf, np.ceil(W).astype(int))
        else:
            W = -math.inf

    return W

def get_V_diff_W(s, a, V1, V2, V_i, C, lamb, s_succ):
    V_diff = V1[V_i[s]] - np.sum(
        np.fromiter(
            (p * np.exp(lamb * C(s, a)) * V2[V_i[s_]]
             for s_, p in s_succ.items()), dtype=float))
    return V_diff

def can_improve(V, V_i, s, a, C, lamb, succ_states):
    V_diff = get_V_diff_W(s, a, V, V, V_i, C, lamb, succ_states[s, a])

    return V_diff < 0, V_diff

def get_w_reachable(s, V_risk, V_i, P, pi_risk, goal, A, C, lamb, k_g, succ_states, visited=None, W_s=None):
    if check_goal(utils.from_literals(s), goal):
        return {s: float('-inf')}

    # Setup
    # =================================================
    W_s = W_s or {}

    visited = visited if visited else set()
    visited.add(s)
    # a_opt = pi_risk[s]

    # reachable = map(lambda _s: _s, mdp_obj[s]['Adj'])
    # reachable_by_actions = {_s['name']: _s['A'] for _s in mdp_obj[s]['Adj']}
    reachable_by_actions = {}
    for a in A:
        for s_ in succ_states[s, a]:
            if s_ not in reachable_by_actions:
                reachable_by_actions[s_] = set()
            reachable_by_actions[s_].add(a)

    # =================================================

    # Find state-action pairs that can improve expected utility
    # =================================================
    candidates = []
    for a in A:
        can, V_diff = can_improve(V_risk, V_i, s, a, C, lamb, succ_states)
        if can:
            candidates.append((a, V_diff))
    # =================================================

    # Find max cost for current state and recursively
    #   compute costs for reachable states
    # =================================================
    W_a = [W(s, a, V_diff, V_i, P, k_g, lamb, succ_states)
           for a, V_diff in candidates]

    W_max_a = max(W_a, default=float('-inf'))
    W_reachable = []
    for s_, A_ in reachable_by_actions.items():
        if s_ not in visited and s_ not in W_s:
            W_s = {
                **W_s,
                **get_w_reachable(
                    s_, V_risk, V_i, P, pi_risk, goal, A, C, lamb, k_g, succ_states, visited, W_s)
            }
            # Go through each action a that can lead to s_
            #   and get maximum difference between W_s[s_] and C(s, a)
            W_reachable.append(
                max([W_s[s_] - C(s, a) for a in A_])
            )

    W_reachable_max = max(W_reachable, default=float('-inf'))

    W_s[s] = W_max_a \
        if (W_max_a >= W_reachable_max) \
        else W_reachable_max

    return W_s


def __get_cmax_reachable(V_risk, P, pi_risk, goal, A, C, lamb, k_g, W_s, succ_states):
    C_maxs_s = {}

    for s in W_s:
        if check_goal(utils.from_literals(s), goal):
            if s not in C_maxs_s:
                C_maxs_s[s] = float('-inf')
            continue
        #reachable_by_actions = {_s['name']: _s['A']
        #                        for _s in mdp_obj[s]['Adj']}
        reachable_by_actions = {}
        for a in A:
            for s_ in succ_states[s, a]:
                if s_ not in reachable_by_actions:
                    reachable_by_actions[s_] = set()
                reachable_by_actions[s_].add(a)

        w_s = W_s[s]
        C_max_reachable = []
        for s_, A_ in reachable_by_actions.items():
            # Go through each action a that can lead to s_
            #   and get maximum difference between W_s[s_] and C(s, a)
            C_max_reachable.append(
                max([W_s[s_] - C(s, a) for a in A_])
            )
        C_max_reachable_max = max(C_max_reachable, default=float('-inf'))

        C_maxs_s[s] = w_s \
            if (w_s >= C_max_reachable_max) \
            else C_max_reachable_max

    return C_maxs_s

def get_cmax_reachable(s, V_risk, V_i, P, pi_risk, goal, A, C, lamb, k_g, succ_states, visited=None, W_s=None):
    if check_goal(utils.from_literals(s), goal):
        return {s: -float('-inf')}
    W_s = W_s or {}
    i = 0
    W_s = get_w_reachable(
        s, V_risk, V_i, P, pi_risk, goal, A, C, lamb, k_g, succ_states, W_s=W_s)
    old_C_maxs_s = {}
    C_maxs_s = W_s
    while old_C_maxs_s != C_maxs_s:
        old_C_maxs_s = C_maxs_s
        C_maxs_s = __get_cmax_reachable(
            V_risk, P, pi_risk, goal, A, C, lamb, k_g, C_maxs_s, succ_states)
        i += 1
    return C_maxs_s, W_s

