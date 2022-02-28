#!/usr/bin/env python3
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
import tensorflow as tf
import argparse
from tensorflow.keras.models import load_model
from nncg.nncg import NNCG

import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D

import sys
sys.path.insert(0,'..')

import config as cfg

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser(description='Train the network given ')

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('-s', '--simulation', action='store_true',
                        help='Entrainer pour l\'environnement de la simulation.')
    action.add_argument('-r', '--robot', action='store_true',
                        help='Entrainer pour l\'environnement des robots.')
    action.add_argument('-d', '--dummy', action='store_true',
                        help='Dummy model.')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('-u', '--upper', action='store_true',
                        help='Entrainer pour la camera du haut.')
    action.add_argument('-l', '--lower', action='store_true',
                        help='Entrainer pour la camera du bas.')

    return parser.parse_args()

def set_config(args):
    if args.upper:
        cfg.camera = "upper"
    else:
        cfg.camera = "lower"
    if args.simulation:
        env = "Simulation"
    else:
        env = "Genere"
    return env

def create_dummy_db():
    return [np.zeros((120, 160, 3))]

def dummy_model():
    inputs = keras.Input(shape=(120, 160, 3))
    x = Conv2D(15, (3, 3), (2, 2), padding='same', groups=3, activation='sigmoid')(inputs)
    return keras.Model(inputs=inputs, outputs=x)

def main():
    args = parse_args()
    if args.dummy:
        model_path = 'dummy_model3'
        model = dummy_model()
        model.summary()
        res = model(np.ones((1, 120, 160, 3)) / 2)
        breakpoint()
    else:
        env = set_config(args)
        model_path = cfg.get_modele_path(env)
        model = load_model(model_path, compile=False)
        model.summary()

    code_path = "."

    
    images = create_dummy_db()

    sse_generator = NNCG()
    sse_generator.keras_compile(images, model, code_path, model_path.strip('.h5'), arch="general", testing=-1)

if __name__ == '__main__':
    main()