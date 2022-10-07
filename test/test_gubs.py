import math
import os
import unittest
import pytest
import numpy as np
import gubs
import mdp
import utils
from pddlgym.structs import Predicate, Type
from pddlgym.inference import check_goal

location_type = Type('location')
direction_type = Type('direction')
robot_type = Type('robot')
robot_at_predicate = Predicate('robot-at', 2, [robot_type, location_type])
move_operator_pred = Predicate('move', 1, [direction_type])

# ==============================================
dir_path = os.path.dirname(os.path.realpath(__file__))
# River 1st problem
env_river, problem = utils.create_problem_instance_from_file(
    dir_path, 'river-alt', problem_index=0)
goal_river_0 = problem.goal

conn_predicates_river_0 = utils.get_literals_that_start_with(problem.initial_state, 'conn')
isriver_predicates_river_0 = utils.get_literals_that_start_with(problem.initial_state, 'is-river')
iswaterfall_predicates_river_0 = utils.get_literals_that_start_with(problem.initial_state, 'is-waterfall')
isbank_predicates_river_0 = utils.get_literals_that_start_with(problem.initial_state, 'is-bank')
isbridge_predicates_river_0 = utils.get_literals_that_start_with(problem.initial_state, 'is-bridge')
isgoal_predicates_river_0 = utils.get_literals_that_start_with(problem.initial_state, 'is-goal')


river_0_location_objs = utils.get_objects_by_name(problem.objects, 'location')
river_0_location_objs = sorted(
    utils.get_objects_by_name(
        problem.objects, 'location'),
        key=lambda s: tuple(reversed(utils.get_coord_from_location_obj(s))))
base_literals = frozenset({
    *conn_predicates_river_0, *isriver_predicates_river_0,
    *iswaterfall_predicates_river_0, *isbank_predicates_river_0,
    *isbridge_predicates_river_0, *isgoal_predicates_river_0
})

river_0_states = utils.create_states_from_base_literals(
    base_literals, [
        frozenset({robot_at_predicate('robot0', obj.name)}) for obj in river_0_location_objs
    ], problem, only_literals=True)

init_river_0, _ = env_river.reset()
A_river_0 = np.array(
            list(env_river.action_space.all_ground_literals(init_river_0)))


reach_river_0 = mdp.get_all_reachable(init_river_0, A_river_0, env_river)
S_river_0 = list(sorted([s for s in reach_river_0], key = lambda s: tuple(reversed(utils.get_coord_from_state(utils.from_literals(s))))))
V_i_river_0 = {s: i for i, s in enumerate(S_river_0)}
succ_states_river_0 = {s: {} for s in reach_river_0}
for s in reach_river_0:
    for a in A_river_0:
        succ_states_river_0[s, a] = reach_river_0[s][a]

# River 2nd problem
env_river.fix_problem_index(1)
problem = env_river.problems[1]
goal_river_1 = problem.goal

conn_predicates_river_1 = utils.get_literals_that_start_with(problem.initial_state, 'conn')
isriver_predicates_river_1 = utils.get_literals_that_start_with(problem.initial_state, 'is-river')
iswaterfall_predicates_river_1 = utils.get_literals_that_start_with(problem.initial_state, 'is-waterfall')
isbank_predicates_river_1 = utils.get_literals_that_start_with(problem.initial_state, 'is-bank')
isbridge_predicates_river_1 = utils.get_literals_that_start_with(problem.initial_state, 'is-bridge')
isgoal_predicates_river_1 = utils.get_literals_that_start_with(problem.initial_state, 'is-goal')

base_literals = frozenset({
    *conn_predicates_river_1, *isriver_predicates_river_1,
    *iswaterfall_predicates_river_1, *isbank_predicates_river_1,
    *isbridge_predicates_river_1, *isgoal_predicates_river_1
})

river_1_location_objs = sorted(
    utils.get_objects_by_name(
        problem.objects, 'location'),
        key=lambda s: tuple(reversed(utils.get_coord_from_location_obj(s))))

river_1_states = utils.create_states_from_base_literals(
    base_literals, [
        frozenset({robot_at_predicate('robot0', obj.name)}) for obj in river_1_location_objs
    ], problem, only_literals=True)
init_river_1, _ = env_river.reset()
A_river_1 = np.array(
            list(env_river.action_space.all_ground_literals(init_river_1)))
reach_river_1 = mdp.get_all_reachable(init_river_1, A_river_1, env_river)
S_river_1 = list(sorted([s for s in reach_river_1], key = lambda s: tuple(reversed(utils.get_coord_from_state(utils.from_literals(s))))))
V_i_river_1 = {s: i for i, s in enumerate(S_river_1)}
succ_states_river_1 = {s: {} for s in reach_river_1}
for s in reach_river_1:
    for a in A_river_1:
        succ_states_river_1[s, a] = reach_river_1[s][a]

# ==============================================

V_risk_river_4x4 = [
    0.0497870683678639, 0.0820849986238988, 0.1353352832366127, 0.2231301601484298,
    0.0301973834223185, 0.0314893483004989, 0.1874293345246811, 0.3678794411714423,
    0.0183156388887342, 0.0066653979229454, 0.2207276647028654, 0.6065306597126334,
    0., 0., 0., 1.,
]
pi_risk_river_4x4 = {
    river_0_states[0]: move_operator_pred('right'), river_0_states[1]: move_operator_pred('right'),
    river_0_states[2]: move_operator_pred('right'), river_0_states[3]: move_operator_pred('down'),
    river_0_states[4]: move_operator_pred('up'), river_0_states[5]: move_operator_pred('up'),
    river_0_states[6]: move_operator_pred('right'), river_0_states[7]: move_operator_pred('down'),
    river_0_states[8]: move_operator_pred('up'), river_0_states[9]: move_operator_pred('left'),
    river_0_states[10]: move_operator_pred('right'), river_0_states[11]: move_operator_pred('down'),
    river_0_states[12]: move_operator_pred('down'), river_0_states[13]: move_operator_pred('down'),
    river_0_states[14]: move_operator_pred('down'), river_0_states[15]: None,
}
P_river_4x4 = [1., 1., 1., 1., 1., 0.84, 0.84,
               1., 1., 0.6, 0.6, 1., 0., 0., 0., 1.]

V_risk_river_5x8 = [
    0.3328710836980793, 0.3678794411714421, 0.4065696597405989,
    0.4493289641172213, 0.4965853037914092, 0.3011942119122019,
    0.2791507712970741, 0.3928587884867178, 0.4945512903870797,
    0.5488116360940262, 0.2725317930340124, 0.2191808473416846,
    0.4755857802539832, 0.5431918049404234, 0.6065306597126332,
    0.2465969639416062, 0.1967830561520015, 0.4992212095003083,
    0.5910034748239898, 0.6703200460356391, 0.2231301601484296,
    0.1738018311606886, 0.4928066938297427, 0.6274195630893582,
    0.7408182206817177, 0.2018965179946552, 0.145506583045073,
    0.4204596961651642, 0.6222873053726429, 0.8187307530779817,
    0.1826835240527344, 0.0991793329329518, 0.2282686130026917,
    0.491238451846789,  0.9048374180359595, 0.,
    0.,                 0.,                 0.,  1.
]

P_river_5x8 = [
    1.,                 1.,                 1.,
    1.,                 1.,                 1.,
    0.99559798358016,   0.9840040243078229, 0.995904,
    1.,                 1.,                 0.98976,
    0.9600100607695572, 0.98976,            1.,
    1.,                 0.9743999999999999, 0.9153852364617343,
    0.9743999999999999, 1.,                 1.,
    0.9359999999999999, 0.8268630911543358, 0.9359999999999999,
    1.,                 1.,                 0.84,
    0.6631578546926016, 0.84,               1.,
    1.,                 0.6,                0.3978946367315038,
    0.6,                1.,                 0.,
    0.,                 0.,                 0.,  1.
]

pi_risk_river_5x8 = {
    river_1_states[0]: move_operator_pred('right'), river_1_states[1]: move_operator_pred('right'), river_1_states[2]: move_operator_pred('right'), river_1_states[3]: move_operator_pred('right'), river_1_states[4]: move_operator_pred('down'),
    river_1_states[5]: move_operator_pred('up'), river_1_states[6]: move_operator_pred('up'), river_1_states[7]: move_operator_pred('up'), river_1_states[8]: move_operator_pred('right'), river_1_states[9]: move_operator_pred('down'),
    river_1_states[10]: move_operator_pred('up'), river_1_states[11]: move_operator_pred('left'), river_1_states[12]: move_operator_pred('right'), river_1_states[13]: move_operator_pred('right'), river_1_states[14]: move_operator_pred('down'),
    river_1_states[15]: move_operator_pred('up'), river_1_states[16]: move_operator_pred('left'), river_1_states[17]: move_operator_pred('right'), river_1_states[18]: move_operator_pred('right'), river_1_states[19]: move_operator_pred('down'),
    river_1_states[20]: move_operator_pred('up'), river_1_states[21]: move_operator_pred('left'), river_1_states[22]: move_operator_pred('right'), river_1_states[23]: move_operator_pred('right'), river_1_states[24]: move_operator_pred('down'),
    river_1_states[25]: move_operator_pred('up'), river_1_states[26]: move_operator_pred('left'), river_1_states[27]: move_operator_pred('right'), river_1_states[28]: move_operator_pred('right'), river_1_states[29]: move_operator_pred('down'),
    river_1_states[30]: move_operator_pred('up'), river_1_states[31]: move_operator_pred('left'), river_1_states[32]: move_operator_pred('right'), river_1_states[33]: move_operator_pred('right'), river_1_states[34]: move_operator_pred('down'),
    river_1_states[35]: move_operator_pred('up'), river_1_states[36]: move_operator_pred('up'), river_1_states[37]: move_operator_pred('up'), river_1_states[38]: move_operator_pred('up'), river_1_states[39]: None
}

def C_factory(goal):
    def C(s, a):
        s = utils.from_literals(s) if type(s) == frozenset else s
        return 0 if check_goal(s, goal) else 1
    return C


class TestGUBS(unittest.TestCase):
    def test_get_cmax_reachable_river_4x4_9(self):
        """
            River 4x4 problem, s0 = '9'
        """
        s0 = init_river_0.literals
        lamb = -0.5
        k_g = 0.01
        C = C_factory(goal_river_0)
        C_maxs_s0, _ = gubs.get_cmax_reachable(
            s0, V_risk_river_4x4, V_i_river_0,
            P_river_4x4, goal_river_0,
            A_river_0, C, lamb, k_g, succ_states_river_0
        )

        expected_cmax_s_values = {
            river_0_states[0]: 6, river_0_states[1]: 7, river_0_states[2]: 6, river_0_states[3]: 5,
            river_0_states[4]: 7, river_0_states[5]: 8, river_0_states[6]: 7, river_0_states[7]: 6,
            river_0_states[8]: 6, river_0_states[9]: 7, river_0_states[10]: 6, river_0_states[11]: 5,
            river_0_states[12]: -math.inf, river_0_states[13]: -math.inf, river_0_states[14]: -math.inf, river_0_states[15]: -math.inf,
        }

        assert len(C_maxs_s0) == len(S_river_0)
        for s in [s0] + S_river_0:
            assert s in C_maxs_s0
            if s in expected_cmax_s_values:
                assert C_maxs_s0[s] == expected_cmax_s_values[s]
            else:
                assert C_maxs_s0[s] == 0

    def test_get_cmax_reachable_river_5x8_7(self):
        """
            River 5x8 problem, s0 = '7'
        """
        s0 = init_river_1.literals
        lamb = -0.1
        k_g = 1
        C = C_factory(goal_river_1)
        W_s0, _ = gubs.get_cmax_reachable(
            s0, V_risk_river_5x8, V_i_river_1,
            P_river_5x8, goal_river_1,
            A_river_1, C, lamb, k_g, succ_states_river_1
        )

        expected_cmax_s_values = {
            river_1_states[0]: 27, river_1_states[1]: 28, river_1_states[2]: 29, river_1_states[3]: 28, river_1_states[4]: 27,
            river_1_states[5]: 28, river_1_states[6]: 29, river_1_states[7]: 30, river_1_states[8]: 29, river_1_states[9]: 28,
            river_1_states[10]: 27, river_1_states[11]: 28, river_1_states[12]: 29, river_1_states[13]: 28, river_1_states[14]: 27,
            river_1_states[15]: 26, river_1_states[16]: 27, river_1_states[17]: 28, river_1_states[18]: 27, river_1_states[19]: 26,
            river_1_states[20]: 25, river_1_states[21]: 26, river_1_states[22]: 27, river_1_states[23]: 26, river_1_states[24]: 25,
            river_1_states[25]: 24, river_1_states[26]: 25, river_1_states[27]: 26, river_1_states[28]: 25, river_1_states[29]: 24,
            river_1_states[30]: 23, river_1_states[31]: 24, river_1_states[32]: 25, river_1_states[33]: 24, river_1_states[34]: 23,
            river_1_states[35]: -math.inf,  river_1_states[36]: -math.inf,  river_1_states[37]: -math.inf,  river_1_states[38]: -math.inf,  river_1_states[39]: -math.inf
        }

        assert len(W_s0) == 40
        for s in river_1_states:
            assert s in W_s0
            if s in expected_cmax_s_values:
                assert W_s0[s] == expected_cmax_s_values[s]
            else:
                assert W_s0[s] == 0
