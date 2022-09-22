import os
import json

naovaCodePath = '../NaovaCode'

camera = 'upper' # doit etre dans {'upper', 'lower'}
training = 'balles' # doit etre dans {'balles', 'robots'}

def get_modele_path(env='Simulation'):
    env = env.lower()
    if env == 'genere':
        env = 'robot'
    return f'modele_{env}_{camera}.h5'
