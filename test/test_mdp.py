import os
import unittest
from copy import deepcopy
import gubs
import mdp
import utils
import numpy as np
import pytest
from itertools import chain
from pddlgym.inference import check_goal
from pddlgym.structs import Predicate, State, Type


def h_1(s):
    return 1

# Define test data for test_domain
# =================================

dir_path = os.path.dirname(os.path.realpath(__file__))
env_test_domain, problem = utils.create_problem_instance_from_file(dir_path, 'test_domain')
goal_test_domain = problem.goal

location_type = Type('location')
robot_at_predicate = Predicate('robot-at', 1, [location_type])
is_goal_predicate = Predicate('is-goal', 1, [location_type])
conn1_predicate = Predicate('conn1', 2, [location_type, location_type])
conn2_predicate = Predicate('conn2', 2, [location_type, location_type])
move1_operator_pred = Predicate('move1', 0)
move2_operator_pred = Predicate('move2', 0)

base_state_literals = frozenset({
    conn1_predicate('s1', 's2'),
    conn1_predicate('s2', 's3'),
    conn2_predicate('s1', 's3'),
    is_goal_predicate('s3')
})

s1, s2, s3, de = utils.create_states_from_base_literals(
    base_state_literals, [
        frozenset({robot_at_predicate('s1')}),
        frozenset({robot_at_predicate('s2')}),
        frozenset({robot_at_predicate('s3')}),
        frozenset({robot_at_predicate('de')})
    ], problem)
# =================================

# Define test data for test_domain_gridworld
# =================================

env_test_domain_gridworld, problem = utils.create_problem_instance_from_file(
    dir_path, 'test_domain_gridworld')
goal_test_domain_gridworld = problem.goal
problem_gridworld = problem

location_type = Type('location')
direction_type = Type('direction')
robot_at_predicate = Predicate('robot-at', 1, [location_type])
conn_prob_predicate = Predicate('conn-prob', 3,
                                [location_type, location_type, direction_type])
move_operator_pred = Predicate('move', 1, [direction_type])

base_state_literals_gridworld = frozenset({
    conn_prob_predicate('s1', 's2', 'right'),
    conn_prob_predicate('s2', 's1', 'left'),
    conn_prob_predicate('s2', 's3', 'right'),
    conn_prob_predicate('s3', 's2', 'left')
})

s1_gridworld, s2_gridworld, s3_gridworld = utils.create_states_from_base_literals(
    base_state_literals_gridworld, [
        frozenset({robot_at_predicate('s1')}),
        frozenset({robot_at_predicate('s2')}),
        frozenset({robot_at_predicate('s3')})
    ], problem)

env_test_domain_gridworld_2, problem = utils.create_problem_instance_from_file(
    dir_path, 'test_domain_gridworld', problem_index=1)
goal_test_domain_gridworld_2 = problem.goal

conn_predicates_gridworld_2 = frozenset((lit for lit in problem.initial_state if lit.predicate.name.startswith('conn')))
(
    s1_gridworld_2, s2_gridworld_2,
    s3_gridworld_2, s4_gridworld_2,
    s5_gridworld_2, s6_gridworld_2,
    s7_gridworld_2, s8_gridworld_2,
    s9_gridworld_2, s10_gridworld_2,
) = utils.create_states_from_base_literals(
    conn_predicates_gridworld_2, [
        frozenset({robot_at_predicate(f's{i}')}) for i in range(1, 11)
    ], problem)

# =================================

# Define test data for river problem
# =================================
env_river, problem = utils.create_problem_instance_from_file(
    dir_path, 'river-alt', problem_index=2)
goal_river = problem.goal

conn_predicates_river = utils.get_literals_that_start_with(problem.initial_state, 'conn')
isriver_predicates_river = utils.get_literals_that_start_with(problem.initial_state, 'is-river')
iswaterfall_predicates_river = utils.get_literals_that_start_with(problem.initial_state, 'is-waterfall')
isbank_predicates_river = utils.get_literals_that_start_with(problem.initial_state, 'is-bank')
isbridge_predicates_river = utils.get_literals_that_start_with(problem.initial_state, 'is-bridge')
isgoal_predicates_river = utils.get_literals_that_start_with(problem.initial_state, 'is-goal')


river_location_objs = utils.get_objects_by_name(problem.objects, 'location')
river_location_objs = sorted(
    utils.get_objects_by_name(
        problem.objects, 'location'),
        key=lambda s: tuple(reversed(utils.get_coord_from_location_obj(s))))
base_literals = frozenset({
    *conn_predicates_river, *isriver_predicates_river,
    *iswaterfall_predicates_river, *isbank_predicates_river,
    *isbridge_predicates_river, *isgoal_predicates_river
})

robot_at_predicate = Predicate('robot-at', 2, [location_type])
river_states = utils.create_states_from_base_literals(
    base_literals, [
        frozenset({robot_at_predicate('robot0', obj.name)}) for obj in river_location_objs
    ], problem)
init_river, _ = env_river.reset()
A_river = np.array(
            list(env_river.action_space.all_ground_literals(init_river)))


empty_set = set()
reach_river = mdp.get_all_reachable(init_river, A_river, env_river)
S_river = list(sorted([s for s in reach_river], key = lambda s: tuple(reversed(utils.get_coord_from_state(State(s, empty_set, empty_set))))))
V_i_river = {s: i for i, s in enumerate(S_river)}
succ_states_river = {s: {} for s in reach_river}
for s in reach_river:
    for a in A_river:
        succ_states_river[s, a] = reach_river[s][a]
# =================================

# gubs
lamb = -0.1
k_g = 1

graph = {
    s1_gridworld.literals: {
        "goal":
        False,
        "Adj": [{
            "state": s1_gridworld.literals,
            "A": {
                move_operator_pred('up'): 1,
                move_operator_pred('down'): 1,
                move_operator_pred('right'): 0.5,
            }
        }, {
            "state": s2_gridworld.literals,
            "A": {
                move_operator_pred('right'): 0.5
            }
        }]
    },
    s2_gridworld.literals: {
        "goal":
        False,
        "Adj": [{
            "state": s2_gridworld.literals,
            "A": {
                move_operator_pred('up'): 1,
                move_operator_pred('down'): 1,
                move_operator_pred('right'): 0.5,
            }
        }, {
            "state": s3_gridworld.literals,
            "A": {
                move_operator_pred('right'): 0.5
            }
        }]
    },
    s3_gridworld.literals: {
        "goal":
        True,
        "Adj": [{
            "state": s3_gridworld.literals,
            "A": {
                move_operator_pred('up'): 1,
                move_operator_pred('down'): 1,
                move_operator_pred('right'): 1,
            }
        }]
    },
}

bpsg = {
    s1_gridworld.literals: {
        "Adj": []
    },
}

explicit_graph_test_dual_criterion = {
    s1.literals: {
        "value": 1,
        "prob": 0,
        "solved": False,
        "expanded": True,
        "pi": "None",
        "Q_v": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Q_p": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Adj": [
            {
                "state": s2.literals,
                "A": {
                    move1_operator_pred(): 1
                }
            },
            {
                "state": s3.literals,
                "A": {
                    move2_operator_pred(): 0.95
                }
            },
            {
                "state": de.literals,
                "A": {
                    move2_operator_pred(): 0.05
                }
            },
        ],
    },
    s2.literals: {
        "value": 1,
        "prob": 1,
        "solved": False,
        "pi": None,
        "expanded": False,
        "Q_v": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Q_p": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Adj": [],
    },
    s3.literals: {
        "value": 1,
        "prob": 1,
        "solved": False,
        "pi": None,
        "expanded": False,
        "Q_v": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Q_p": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Adj": [],
    },
    de.literals: {
        "value": 1,
        "prob": 1,
        "solved": False,
        "pi": None,
        "expanded": False,
        "Q_v": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Q_p": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Adj": [],
    }
}

bpsg_test_dual_criterion = {
    s1.literals: {
        "Adj": [],
    }
}

bpsg_2 = {
    s1_gridworld.literals: {
        "Adj": [
            {
                "state": s1_gridworld.literals,
                "A": {
                    move_operator_pred('right'): 0.5,
                }
            },
            {
                "state": s2_gridworld.literals,
                "A": {move_operator_pred('right'): 0.5}
            }
        ]
    },
    s2_gridworld.literals: {
        "Adj": [
            {
                "state": s2_gridworld.literals,
                "A": {
                    move_operator_pred('right'): 0.5,
                }
            },
            {
                "state": s3_gridworld.literals,
                "A": {move_operator_pred('right'): 0.5}
            }
        ]
    },
    s3_gridworld.literals: {
        "Adj": [{
            "state": s3_gridworld.literals,
            "A": {
                move_operator_pred('up'): 1,
            }
        }]
    },
}

explicit_graph_test_dual_criterion_2 = {
    s1.literals: {
        "value": np.exp(lamb),
        "prob": 1,
        "solved": False,
        "expanded": True,
        "pi": move1_operator_pred(),
        "Q_v": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Q_p": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Adj": [
            {
                "state": s2.literals,
                "A": {
                    move1_operator_pred(): 1
                }
            },
            {
                "state": s3.literals,
                "A": {
                    move2_operator_pred(): 0.95
                }
            },
            {
                "state": de.literals,
                "A": {
                    move2_operator_pred(): 0.05
                }
            },
        ],
    },
    de.literals: {
        "value": 1,
        "prob": 0,
        "solved": False,
        "pi": None,
        "expanded": False,
        "Q_v": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Q_p": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Adj": [{
            "state": de.literals,
            "A": {
                move1_operator_pred(): 1
            }
        }],
    },
    s2.literals: {
        "value": 1,
        "prob": 1,
        "solved": False,
        "pi": None,
        "expanded": False,
        "Q_v": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Q_p": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Adj": [],
    },
    s3.literals: {
        "value": 1,
        "prob": 1,
        "solved": False,
        "pi": None,
        "expanded": False,
        "Q_v": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Q_p": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Adj": [],
    }
}

explicit_graph_test_dual_criterion_3 = {
    s1.literals: {
        "value": np.exp(lamb),
        "prob": 1,
        "solved": False,
        "expanded": True,
        "pi": "a",
        "Q_v": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Q_p": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Adj": [
            {
                "state": s2.literals,
                "A": {
                    move1_operator_pred(): 1
                }
            },
            {
                "state": s3.literals,
                "A": {
                    move2_operator_pred(): 0.95
                }
            },
            {
                "state": de.literals,
                "A": {
                    move2_operator_pred(): 0.05
                }
            },
        ],
    },
    de.literals: {
        "value": 0.001,
        "prob": 0,
        "solved": False,
        "pi": None,
        "expanded": False,
        "Q_v": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Q_p": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Adj": [{
            "state": de.literals,
            "A": {
                move1_operator_pred(): 1
            }
        }],
    },
    s2.literals: {
        "value": 0,
        "prob": 1,
        "solved": False,
        "pi": None,
        "expanded": True,
        "Q_v": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Q_p": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Adj": [{
            "state": s3.literals,
            "A": {
                move1_operator_pred(): 1
            }
        }],
    },
    s3.literals: {
        "value": 1,
        "prob": 1,
        "solved": False,
        "pi": None,
        "expanded": False,
        "Q_v": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Q_p": {
            move1_operator_pred(): 1,
            move2_operator_pred(): 1
        },
        "Adj": [],
    }
}

bpsg_extended = {
    (s1_gridworld.literals, 0): {"Adj": []},
}

bpsg_2_extended = {
    (s1_gridworld.literals, 0): {
        "Adj": [
            {
                "state": (s1_gridworld.literals, 1),
                "A": {
                    move_operator_pred('right'): 0.5,
                }
            },
            {
                "state": (s2_gridworld.literals, 1),
                "A": {move_operator_pred('right'): 0.5}
            }
        ]
    },
    (s2_gridworld.literals, 1): {
        "Adj": [
            {
                "state": (s2_gridworld.literals, 2),
                "A": {
                    move_operator_pred('right'): 0.5,
                }
            },
            {
                "state": (s3_gridworld.literals, 2),
                "A": {move_operator_pred('right'): 0.5}
            }
        ]
    },
    (s3_gridworld.literals, 2): {
        "Adj": [{
            "state": (s3_gridworld.literals, 2),
            "A": {
                move_operator_pred('up'): 1,
            }
        }]
    },
}

bpsg_3_extended = {
    (s1_gridworld.literals, 0): {
        "Adj": [
            {
                "state": (s1_gridworld.literals, 1),
                "A": {
                    move_operator_pred('right'): 0.5,
                }
            },
            {
                "state": (s2_gridworld.literals, 1),
                "A": {move_operator_pred('right'): 0.5}
            }
        ]
    },
    (s2_gridworld.literals, 1): {
        "Adj": [
            {
                "state": (s2_gridworld.literals, 2),
                "A": {
                    move_operator_pred('right'): 0.5,
                }
            }
        ]
    },
    (s3_gridworld.literals, 2): {"Adj": []},
}

explicit_graph_extended = {
    (s1_gridworld.literals, 0): {
        "value": 0.6826277,
        "prob": 1,
        "solved": True,
        "expanded": False,
        "pi": None,
        "Adj": []
    },
}

explicit_graph_extended_expanded = {
    (s1_gridworld.literals, 0): {
        "value": 0.6826277,
        "prob": 1,
        "solved": False,
        "expanded": True,
        "pi": None,
        "Adj": [
            {
                "state": (s1_gridworld.literals, 1),
                "A": {
                    move_operator_pred('up'): 1,
                    move_operator_pred('down'): 1,
                    move_operator_pred('right'): 0.5,
                }
            },
            {
                "state": (s2_gridworld.literals, 1),
                "A": {move_operator_pred('right'): 0.5}
            }
        ]
    },
    (s1_gridworld.literals, 1): {
        "value": 0.6826277,
        "prob": 1,
        "solved": True,
        "expanded": False,
        "pi": None,
        "Adj": []
    },
    (s2_gridworld.literals, 1): {
        "value": 0.82621287,
        "prob": 1,
        "solved": True,
        "expanded": False,
        "pi": None,
        "Adj": []
    }
}

bpsg_river_test = {
    (river_states[3].literals, 0): {
        'Adj': [
            {'state': (river_states[3].literals, 1), 'A': {'W': 1}}
        ]
    },
    (river_states[3].literals, 1): {'Adj': []}
}


explicit_graph_river_test = {
    (river_states[3].literals, 0): {
        "value": 0.2207276647028654,
        "prob": 0.6,
        "solved": False,
        "expanded": True,
        "pi": move_operator_pred('right'),
        "Adj": [
            {"state": (river_states[3].literals, 1), "A": {move_operator_pred('left'): 1}},
            {"state": (river_states[4].literals, 1), "A": {move_operator_pred('right'): 1}},
            {"state": (river_states[0].literals, 1), "A": {move_operator_pred('up'): 1}},
        ],
    },
    (river_states[3].literals, 1): {
        "solved": False,
        "value": 0.2207276647028654,
        "prob": 0.6,
        "pi": move_operator_pred('right'),
        "expanded": True,
        "Adj": [
            {"state": (river_states[3].literals, 2), "A": {move_operator_pred('left'): 1}},
            {"state": (river_states[4].literals, 2), "A": {move_operator_pred('right'): 1}},
            {"state": (river_states[0].literals, 2), "A": {move_operator_pred('up'): 1}},
        ],
    },
    (river_states[4].literals, 1): {
        "solved": True,
        "value": 0.36391839582758007,
        "prob": 0.6,
        "pi": None,
        "expanded": False,
        "Adj": [],
    },
    (river_states[0].literals, 1): {
        "solved": True,
        "value": 0.22313016014842985,
        "prob": 1.0,
        "pi": None,
        "expanded": False,
        "Adj": [],
    },
    (river_states[3].literals, 2): {
        "solved": True,
        "value": 0.1353352832366127,
        "prob": 1.0,
        "pi": None,
        "expanded": False,
        "Adj": [],
    },
    (river_states[4].literals, 2): {
        "solved": True,
        "value": 0.36391839582758007,
        "prob": 0.6,
        "pi": None,
        "expanded": False,
        "Adj": [],
    },
    (river_states[0].literals, 2): {
        "solved": True,
        "value": 0.22313016014842985,
        "prob": 1.0,
        "pi": None,
        "expanded": False,
        "Adj": [],
    },
}

graph_ex = {
    'u': {'Adj': [
        {'state': 'v'},
        {'state': 'x'},
    ]},
    'v': {'Adj': [{'state': 'y'}]},
    'w': {'Adj': [{'state': 'y'}, {'state': 'z'}]},
    'x': {'Adj': [{'state': 'v'}]},
    'y': {'Adj': [{'state': 'x'}]},
    'z': {'Adj': [{'state': 'z'}]},
}


def C_factory(goal):
    def C(s, a):
        s = utils.from_literals(s) if type(s) == frozenset else s
        return 0 if check_goal(s, goal) else 1

    return C


C_test_domain = C_factory(goal_test_domain)
C_test_domain_gridworld = C_factory(goal_test_domain_gridworld)
C_river = C_factory(goal_river)

init_gridworld, _ = env_test_domain_gridworld.reset()
A_gridworld = np.array(
            list(env_test_domain_gridworld.action_space.all_ground_literals(init_gridworld)))
reach_gridworld = mdp.get_all_reachable(init_gridworld, A_gridworld, env_test_domain_gridworld)
S_gridworld = list(sorted([s for s in reach_gridworld]))
V_i_gridworld = {s: i for i, s in enumerate(S_gridworld)}

succ_states_gridworld = {s: {} for s in reach_gridworld}
for s in reach_gridworld:
    for a in A_gridworld:
        succ_states_gridworld[s, a] = reach_gridworld[s][a]

V_risk_gridworld, P_risk_gridworld, pi_risk_gridworld, _ = gubs.dual_criterion(
    lamb, V_i_gridworld, S_gridworld, h_1, goal_test_domain_gridworld, succ_states_gridworld, A_gridworld)
C_maxs, _ = gubs.get_cmax_reachable(init_gridworld.literals, V_risk_gridworld, V_i_gridworld, P_risk_gridworld, pi_risk_gridworld, goal_test_domain_gridworld, A_gridworld, C_test_domain_gridworld, lamb, k_g, succ_states_gridworld)

class TestMDPGraph(unittest.TestCase):
    def test_expand_state_dual_criterion(self):

        init_state, *_ = env_test_domain.reset()
        explicit_graph = mdp.add_state_graph(init_state.literals, {})
        explicit_graph[init_state.literals] = {
            "value": 1,
            "prob": 0,
            "solved": False,
            "expanded": False,
            "pi": None,
            **explicit_graph[init_state.literals]
        }

        A = env_test_domain.action_space.all_ground_literals(init_state)
        init_state_neighbours = chain(*[
            mdp.get_successor_states_check_exception(
                init_state, a, env_test_domain.domain, return_probs=False)
            for a in A
        ])
        new_explicit_graph = mdp.expand_state_dual_criterion(
            init_state.literals, h_1, h_1, env_test_domain, explicit_graph,
            goal_test_domain, A)

        assert new_explicit_graph[init_state.literals]['expanded']
        assert new_explicit_graph[init_state.literals]['prob'] == 0

        for s in init_state_neighbours:
            assert s.literals in new_explicit_graph
            assert new_explicit_graph[s.literals]['value'] == 1
            if s == init_state:
                assert new_explicit_graph[s.literals]['prob'] == 0
            else:
                assert new_explicit_graph[s.literals]['prob'] == 1
            assert new_explicit_graph[s.literals]['pi'] == None

    def test_unexpanded_states_1(self):
        init_state, *_ = env_test_domain_gridworld.reset()
        explicit_graph = mdp.add_state_graph(init_state.literals, {})
        explicit_graph[init_state.literals] = {
            "value": 1,
            "prob": 0,
            "solved": False,
            "expanded": False,
            "pi": None,
            **explicit_graph[init_state.literals]
        }

        unexpanded = mdp.get_unexpanded_states(goal_test_domain_gridworld,
                                               explicit_graph, bpsg)
        self.assertListEqual(unexpanded, [s1_gridworld.literals])

    def test_unexpanded_states_2(self):
        init_state, *_ = env_test_domain_gridworld.reset()
        explicit_graph = mdp.add_state_graph(init_state.literals, {})
        explicit_graph[init_state.literals] = {
            "value": 1,
            "prob": 0,
            "solved": False,
            "expanded": False,
            "pi": None,
            **explicit_graph[init_state.literals]
        }
        A = env_test_domain_gridworld.action_space.all_ground_literals(
            init_state)
        # Expand states '1' and '2':
        explicit = mdp.expand_state_dual_criterion(s1_gridworld.literals, h_1, h_1,
                                                   env_test_domain_gridworld,
                                                   explicit_graph,
                                                   goal_test_domain_gridworld,
                                                   A)
        explicit = mdp.expand_state_dual_criterion(s2_gridworld.literals, h_1, h_1,
                                                   env_test_domain_gridworld,
                                                   explicit,
                                                   goal_test_domain_gridworld,
                                                   A)

        unexpanded = mdp.get_unexpanded_states(goal_test_domain_gridworld,
                                               explicit, explicit)
        self.assertListEqual(unexpanded, [])

    def test_update_partial_solution(self):
        explicit_graph = deepcopy(graph)
        for v in explicit_graph.values():
            v['pi'] = None
            v['expanded'] = True
        explicit_graph[s1_gridworld.literals]['pi'] = move_operator_pred('right')

        new_bpsg = mdp.update_partial_solution(s1_gridworld.literals, bpsg,
                                               explicit_graph)

        self.assertDictEqual(
            new_bpsg, {
                s1_gridworld.literals: {
                    "Adj": [{
                        "state": s1_gridworld.literals,
                        "A": {
                            move_operator_pred('right'): 0.5
                        }
                    }, {
                        "state": s2_gridworld.literals,
                        "A": {
                            move_operator_pred('right'): 0.5
                        }
                    }]
                },
                s2_gridworld.literals: {
                    "Adj": []
                },
            })

    def test_update_partial_solution_changes(self):
        expanded_states = [
            s1_gridworld_2.literals, s2_gridworld_2.literals, s3_gridworld_2.literals, s4_gridworld_2.literals
        ]

        A = env_test_domain_gridworld_2.action_space.all_ground_literals(s1_gridworld_2)
        # mdp_ = deepcopy(graph_env_1)
        explicit_graph = mdp.add_state_graph(s1_gridworld_2.literals, {})
        explicit_graph[s1_gridworld_2.literals] = {
            "value": 1,
            "prob": 0,
            "solved": False,
            "expanded": False,
            "pi": None,
            **explicit_graph[s1_gridworld_2.literals]
        }
        explicit_graph = mdp.expand_state_dual_criterion(s1_gridworld_2.literals, h_1, h_1, env_test_domain_gridworld_2, explicit_graph, goal_test_domain_gridworld_2, A)

        # build graph and mark states as expanded
        for i, s in enumerate(expanded_states[1:]):
            #explicit_graph = mdp.add_state_graph(s, explicit_graph)
            explicit_graph = mdp.expand_state_dual_criterion(s, h_1, h_1, env_test_domain_gridworld_2, explicit_graph, goal_test_domain_gridworld_2, A)
            explicit_graph[s]['expanded'] = True

        S_ = [
            s1_gridworld_2.literals, s2_gridworld_2.literals, s3_gridworld_2.literals, s4_gridworld_2.literals,
            s5_gridworld_2.literals, s6_gridworld_2.literals, s7_gridworld_2.literals, s8_gridworld_2.literals,
            s9_gridworld_2.literals, s10_gridworld_2.literals
        ]
        V_i = {S_[i]: i for i in range(len(S_))}
        pi = [
            move_operator_pred('down'),
            move_operator_pred('right'),
            move_operator_pred('right'),
            move_operator_pred('right'), None, None, None, None, None, None
        ]

        #explicit_graph = deepcopy(mdp_)
        for s in explicit_graph:
            explicit_graph[s]['pi'] = pi[V_i[s]]

        bpsg_ = {
            s1_gridworld_2.literals: {
                'Adj': [{
                    'A': {
                        move_operator_pred('right'): 0.5
                    },
                    'state': s1_gridworld_2.literals
                }, {
                    'A': {
                        move_operator_pred('right'): 0.5
                    },
                    'state': s2_gridworld_2.literals
                }]
            },
            s2_gridworld_2.literals: {
                'Adj': [{
                    'A': {
                        move_operator_pred('right'): 0.5
                    },
                    'state': s2_gridworld_2.literals
                }, {
                    'A': {
                        move_operator_pred('right'): 0.5
                    },
                    'state': s3_gridworld_2.literals
                }]
            },
            s3_gridworld_2.literals: {
                'Adj': [{
                    'A': {
                        move_operator_pred('right'): 0.5
                    },
                    'state': s3_gridworld_2.literals
                }, {
                    'A': {
                        move_operator_pred('right'): 0.5
                    },
                    'state': s4_gridworld_2.literals
                }]
            },
            s4_gridworld_2.literals: {
                'Adj': []
            }
        }

        new_bpsg = mdp.update_partial_solution(s1_gridworld_2.literals, bpsg_,
                                                     explicit_graph)

        self.assertDictEqual(
            new_bpsg, {
                s1_gridworld_2.literals: {
                    'Adj': [{
                        'A': {
                            move_operator_pred('down'): 0.5
                        },
                        'state': s1_gridworld_2.literals
                    }, {
                        'A': {
                            move_operator_pred('down'): 0.5
                        },
                        'state': s6_gridworld_2.literals
                    }]
                },
                s6_gridworld_2.literals: {
                    'Adj': []
                }
            })

    def test_value_iteration_dual_criterion(self):
        Z = [s1.literals]

        obs, *_ = env_test_domain.reset()
        epsilon = 1e-3
        A = np.array(
            list(env_test_domain.action_space.all_ground_literals(obs)))
        new_explicit_graph_, converged, *_ = mdp.value_iteration_dual_criterion(
            explicit_graph_test_dual_criterion, bpsg_test_dual_criterion, A, Z,
            goal_test_domain, lamb, C_test_domain, epsilon)

        assert converged

        assert new_explicit_graph_[s1.literals]['value'] == np.exp(lamb)
        assert new_explicit_graph_[s1.literals]['prob'] == 1
        assert new_explicit_graph_[s1.literals]['pi'] == move1_operator_pred(
        ) or new_explicit_graph_[s1.literals]['pi'] == move2_operator_pred()

    def test_value_iteration_dual_criterion_2(self):
        """
            expand state '4' after expanding '1'
        """
        Z = [s1.literals, de.literals]

        obs, *_ = env_test_domain.reset()
        epsilon = 1e-3
        A = np.array(
            list(env_test_domain.action_space.all_ground_literals(obs)))
        new_explicit_graph_, converged, *_ = mdp.value_iteration_dual_criterion(
            explicit_graph_test_dual_criterion_2, bpsg_test_dual_criterion, A,
            Z, goal_test_domain, lamb, C_test_domain, epsilon)

        assert converged

        assert new_explicit_graph_[s1.literals]['value'] == np.exp(lamb)
        assert new_explicit_graph_[s1.literals]['prob'] == 1
        assert new_explicit_graph_[s1.literals]['pi'] == move1_operator_pred()

        assert new_explicit_graph_[de.literals]['value'] < epsilon**0.5
        assert new_explicit_graph_[de.literals]['prob'] == 0
        assert new_explicit_graph_[de.literals]['pi'] == move1_operator_pred()

    def test_value_iteration_dual_criterion_3(self):
        """
            expand state '2' after expanding '1'
        """
        Z = [s1.literals, s2.literals]

        obs, *_ = env_test_domain.reset()
        epsilon = 1e-3
        A = np.array(
            list(env_test_domain.action_space.all_ground_literals(obs)))
        new_explicit_graph_, converged, *_ = mdp.value_iteration_dual_criterion(
            explicit_graph_test_dual_criterion_3, bpsg_test_dual_criterion, A,
            Z, goal_test_domain, lamb, C_test_domain, epsilon)
        self.assertAlmostEqual(new_explicit_graph_[s1.literals]['value'],
                               np.exp(lamb * 2))

        assert converged

        assert new_explicit_graph_[s1.literals]['prob'] == 1
        assert new_explicit_graph_[s1.literals]['pi'] == move1_operator_pred()

        assert new_explicit_graph_[s2.literals]['value'] == np.exp(lamb)
        assert new_explicit_graph_[s2.literals]['prob'] == 1
        assert new_explicit_graph_[s2.literals]['pi'] == move1_operator_pred()

    def test_update_partial_solution_extended(self):

        explicit_ = deepcopy(explicit_graph_extended_expanded)
        explicit_[(s1_gridworld.literals, 0)]["pi"] = move_operator_pred('right')

        #new_bpsg = mdp.update_partial_solution(s1_gridworld_2, bpsg_, explicit_graph)
        new_bpsg = mdp.update_partial_solution_extended(
            s1_gridworld.literals, C_river, bpsg_extended, explicit_)

        self.assertDictEqual(new_bpsg, {
            (s1_gridworld.literals, 0): {
                "Adj": [
                    {
                        "state": (s1_gridworld.literals, 1),
                        "A": {move_operator_pred('right'): 0.5}
                    },
                    {
                        "state": (s2_gridworld.literals, 1),
                        "A": {move_operator_pred('right'): 0.5}
                    }
                ]
            },
            (s2_gridworld.literals, 1): {"Adj": []},
            (s1_gridworld.literals, 1): {"Adj": []},
        })

    def test_update_partial_solution_changes_extended(self):
        #not_extended_expanded_states = [river_states[0], river_states[3], river_states[4]]
        #not_extended_solved_states = [river_states[3]]

        #mdp_obj = deepcopy(graph_env_river_test)
        explicit_ = deepcopy(explicit_graph_river_test)
        # mark states as expanded
        #for s in not_extended_expanded_states:
        #    explicit_[s]['expanded'] = True
        #for s in not_extended_solved_states:
        #    explicit_[s]['solved'] = True

        new_bpsg = mdp.update_partial_solution_extended(river_states[3].literals, C_river, bpsg_river_test, explicit_)

        self.assertDictEqual(new_bpsg, {
            (river_states[3].literals, 0): {
                'Adj': [
                    {'state': (river_states[4].literals, 1), 'A': {move_operator_pred('right'): 1}}
                ]
            },
            (river_states[4].literals, 1): {'Adj': []}
        })



    #def test_unexpanded_states_extended_1(self):
    #    explicit_graph_extended_ = deepcopy(explicit_graph_extended)
    #    explicit_graph_extended_[(s1_gridworld, 0)]["solved"] = False

    #    unexpanded = mdp.get_unexpanded_states_extended(
    #        goal_test_domain_gridworld, explicit_graph_extended_, bpsg_extended)
    #    self.assertListEqual(unexpanded, [(s1_gridworld, 0)])

    #def test_unexpanded_states_extended_2(self):
    #    # Expand states '1' and '2':
    #    explicit_ = mdp.expand_state_gubs(
    #        (s1_gridworld, 0), env_test_domain_gridworld, goal_test_domain_gridworld, explicit_graph_extended, C_test_domain_gridworld, C_maxs, V_risk_gridworld, P_risk_gridworld, pi_risk_gridworld, V_i_gridworld, A_gridworld)
    #    explicit_ = mdp.expand_state_gubs(
    #        (s2_gridworld, 1), env_test_domain_gridworld, goal_test_domain_gridworld, explicit_, C_test_domain_gridworld, C_maxs, V_risk_gridworld, P_risk_gridworld, pi_risk_gridworld, V_i_gridworld, A_gridworld)

    #    unexpanded = mdp.get_unexpanded_states_extended(
    #        goal_test_domain_gridworld, explicit_, bpsg_2)
    #    self.assertListEqual(unexpanded, [])

    def test_unexpanded_states_extended_v2(self):
        # Expand states '1' and '2':
        pi_risk_ = {S_gridworld[i]: a for i, a in enumerate(pi_risk_gridworld)}
        explicit_, C_maxs_, *_ = mdp.expand_state_gubs_v2(
            (s1_gridworld.literals, 0), h_1, env_test_domain_gridworld, goal_test_domain_gridworld, explicit_graph_extended, C_test_domain_gridworld, C_maxs, V_risk_gridworld, P_risk_gridworld, pi_risk_, V_i_gridworld, A_gridworld, k_g, lamb)
        explicit_, *_ = mdp.expand_state_gubs_v2(
            (s2_gridworld.literals, 1), h_1, env_test_domain_gridworld, goal_test_domain_gridworld, explicit_, C_test_domain_gridworld, C_maxs_, V_risk_gridworld, P_risk_gridworld, pi_risk_, V_i_gridworld, A_gridworld, k_g, lamb)

        unexpanded = mdp.get_unexpanded_states_extended(
            goal_test_domain_gridworld, explicit_, bpsg_2)
        self.assertListEqual(unexpanded, [])

    def test_find_ancestors_extended(self):
        ancestors, _ = mdp.find_ancestors_extended(
            (s1_gridworld.literals, 0), bpsg_extended, C_test_domain_gridworld)

        self.assertListEqual(ancestors, [])

    def test_find_ancestors_2_extended(self):
        # Test for second bpsg example
        ancestors_, _ = mdp.find_ancestors_extended(
            (s3_gridworld.literals, 2), bpsg_2_extended, C_test_domain_gridworld)
        ancestors = set(ancestors_)
        self.assertSetEqual(ancestors, set([(s1_gridworld.literals, 0), (s2_gridworld.literals, 1)]))

    def test_find_ancestors_3_extended(self):
        # Test for third bpsg example
        ancestors_, _ = mdp.find_ancestors_extended(
            (s2_gridworld.literals, 1), bpsg_3_extended, C_test_domain_gridworld)
        ancestors = set(ancestors_)
        self.assertSetEqual(ancestors, set([(s1_gridworld.literals, 0)]))


    def test_dfs(self):
        d, f, colors = mdp.dfs(graph)

        self.assertListEqual(colors, ['b'] * len(graph))
        self.assertListEqual(d, [1, 2, 3])
        self.assertListEqual(f, [6, 5, 4])

    def test_dfs_2(self):
        # No need to define the 'A' dict for each state
        #   because it is only a graph to test dfs,
        #   not mdp functinoality

        trace = []
        d, f, colors = mdp.dfs(graph_ex, on_finish=lambda s, *_: trace.append(s))

        self.assertListEqual(colors, ['b', 'b', 'b', 'b', 'b', 'b'])
        self.assertListEqual(d, [1, 2, 9, 4, 3, 10])
        self.assertListEqual(f, [8, 7, 12, 5, 6, 11])
        self.assertListEqual(trace, ['x', 'y', 'v', 'u', 'z', 'w'])

    def test_topological_sort(self):
        topsort = mdp.topological_sort(graph)

        self.assertListEqual(topsort, [s1_gridworld.literals, s2_gridworld.literals, s3_gridworld.literals])

    def test_topological_sort_2(self):

        topsort = mdp.topological_sort(graph_ex)

        self.assertListEqual(topsort, ['w', 'z', 'u', 'v', 'y', 'x'])

    def test_sccs(self):
        expected_sccs = set({
            frozenset({'u'}),
            frozenset({'v', 'x', 'y'}),
            frozenset({'w'}),
            frozenset({'z'})
        })
        sccs = mdp.get_sccs(graph_ex)

        self.assertSetEqual(sccs, expected_sccs)

    def test_sccs_test_domain(self):
        expected_sccs = set({
            frozenset({s1.literals}),
            frozenset({s2.literals}),
            frozenset({s3.literals}),
            frozenset({de.literals})
        })
        sccs = mdp.get_sccs(explicit_graph_test_dual_criterion_3)

        self.assertSetEqual(sccs, expected_sccs)

    def test_sccs_test_domain_2(self):
        de1 = de.literals
        de2 = 'de2' # string just to represent new state
        graph_ = deepcopy(explicit_graph_test_dual_criterion_3)
        graph_[de1]['Adj'] = [{
            "state": de2,
            "A": { move1_operator_pred(): 1 }
        }]
        graph_[de2] = {
            "Adj": [{
                "state": de1,
                "A": {
                    move1_operator_pred(): 1
                }
            }]
        }

        expected_sccs = set({
            frozenset({s1.literals}),
            frozenset({s2.literals}),
            frozenset({s3.literals}),
            frozenset({de1, de2})
        })
        sccs = mdp.get_sccs(graph_)

        self.assertSetEqual(sccs, expected_sccs)
