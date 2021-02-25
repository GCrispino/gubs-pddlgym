import itertools
import os

envs = [
    #('river5-5-40-0.json', 16),
    #('river5-10-40-0.json', 41),
    # ('river5-50-40-0.json', 241)
    ('PDDLEnvRiver-alt-v0', 0),
    ('PDDLEnvRiver-alt-v0', 1),
     #('PDDLEnvRiver-alt-v0', 3),
     #('PDDLEnvTireworld-v0', 0),
     #('PDDLEnvTireworld-v0', 1),
     #('PDDLEnvTireworld-v0', 2)
]
k_gs = [
    #0.01,
    #0.1,
    #0.5,
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

lambs = [
    -0.01,
    -0.05,
    -0.1,
    -0.2,
    -0.3,
    -0.4,
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
    'ao',
    'vi'
]

try:
    for (env, prob_index), k_g, lamb, alg in itertools.product(envs, k_gs, lambs, algorithms):
        os.system(
            f'python src/main.py --env {env} --problem_index {prob_index} --lambda {lamb} --k_g {k_g} --algorithm {alg} --epsilon 1e-10 --render_and_save')
except Exception as e:
    raise e
finally:
    exit()
