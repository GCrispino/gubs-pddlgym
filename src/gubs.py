import itertools
from utils import get_values
import numpy as np
from pddlgym.inference import check_goal

def dual_criterion(lamb, V_i, S, goal, succ_states, A, c=1, epsilon=1e-3, n_iter=None):
    def u(c): return np.exp(lamb * c)

    G_i = [V_i[s] for s in V_i if check_goal(s, goal)]
    not_goal = [s for s in S if not check_goal(s, goal)]
    n_states = len(S)

    # initialize
    V = np.zeros(n_states, dtype=float)
    pi = np.full(n_states, None)
    P = np.zeros(n_states, dtype=float)
    V[G_i] = -np.sign(lamb)
    P[G_i] = 1
    if not isinstance(A, np.ndarray):
        A = np.array(A)

    i = 0

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
                    u(c) * V[V_i[s_]] * p for s_, p in all_reachable[i].items()
                ]) for i in i_A_max_prob
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


def get_cmax(V, V_i, P, S, succ_states, A, lamb, k_g, c=1):
    X = get_X(V, V_i, lamb, S, succ_states, A)
    W = np.zeros(len(X))

    for i, ((s, a), x) in enumerate(X):
        #denominator = k_g * (np.sum(np.fromiter((s_['A'][a] * P[V_i[s_['name']]]
        #                                         for s_ in mdp.find_reachable(s, a, mdp_obj)), dtype=float)) - P[V_i[s]])
        denominator = k_g * (np.sum(np.fromiter((p * P[V_i[s_]]
                                                 for s_, p in succ_states[s, a].items()), dtype=float)) - P[V_i[s]])
        if denominator == 0:
            W[i] = -np.inf
        else:
            W[i] = -(1 / lamb) * np.log(
                x / denominator
            )

    try:
        C_max = np.max(W[np.invert(np.isnan(W))])
    except:
        return 0
    if C_max < 0 or C_max == np.inf:
        return 0

    return int(np.ceil(C_max))

def egubs_vi(V_dual, P_dual, pi_dual, C_max, lamb, k_g, V_i, S, goal, succ_states, A, c=1):
    G_i = [V_i[s] for s in V_i if check_goal(s, goal)]
    n_states = len(S)
    n_actions = len(A)

    V = np.zeros((n_states, C_max + 1))
    V_dual_C = np.zeros((n_states, C_max + 2))
    P = np.zeros((n_states, C_max + 2))
    pi = np.full((n_states, C_max + 2), None)

    V_dual_C[G_i, :] = V_dual[G_i]
    P[G_i, :] = 1
    V_dual_C[:, C_max + 1] = V_dual.T
    P[:, C_max + 1] = P_dual.T
    pi[:, C_max + 1] = pi_dual.T

    n_updates = 0
    for C in reversed(range(C_max + 1)):
        Q = np.zeros(n_actions)
        P_a = np.zeros(n_actions)
        for s in S:
            i_s = V_i[s]
            n_updates += 1
            for i_a, a in enumerate(A):
                #c__ = 0 if mdp_obj[s]['goal'] else c
                c__ = 0 if check_goal(s, goal) else c
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
