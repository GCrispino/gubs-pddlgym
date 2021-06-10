import itertools
import os

envs = [
    ('PDDLEnvRiver-alt-v0', 0),
    ('PDDLEnvRiver-alt-v0', 1),
     #('PDDLEnvRiver-alt-v0', 3),
     #('PDDLEnvTireworld-v0', 0),
     #('PDDLEnvTireworld-v0', 1),
     #('PDDLEnvTireworld-v0', 2)
]

# river
k_gs = [
    0.01,
    0.1,
    0.5,
    1
]

# tireworld
#k_gs = [
#    1e-7,
#    1e-6,
#    1e-5,
#    1e-4,
#    1e-3,
#    1e-2,
#    1e-1,
#    1,
#]

# river
lambs = [
    #-0.01,
    #-0.05,
    -0.1,
    #-0.2,
    #-0.3,
    #-0.4,
    #-0.5
]

# tireworld
#lambs = [
#    #-0.2,
#    #-0.22,
#    #-0.24,
#    #-0.26,
#    #-0.28,
#    -0.3,
#]


algorithms = [
    ('vi', 'vi'),
    ('vi', 'ao'),
    ('lao', 'ao'),
    ('lao_eliminate_traps', 'ao'),
]

try:
    for (env, prob_index), k_g, lamb, (alg_dc, alg_gubs) in itertools.product(envs, k_gs, lambs, algorithms):
        os.system(
            f'python src/main.py --env {env} --problem_index {prob_index} --lambda {lamb} --k_g {k_g} --algorithm_dc {alg_dc} --algorithm_gubs {alg_gubs} --epsilon 1e-10 --render_and_save')
except Exception as e:
    raise e
finally:
    exit()
