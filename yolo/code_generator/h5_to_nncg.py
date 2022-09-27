#!/usr/bin/env python3
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model
from nncg.nncg import NNCG

import yolo.utils.args_parser as args_parser
from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def create_dummy_db():
    return [np.zeros((120, 160, 3))]

def main():
    args = args_parser.parse_args_env_cam('Convertit un modele keras (.h5) en un fichier .cpp reproduisant l\'execution du modele.')
    env = args_parser.set_config(args)
    model_path = cfg_prov.get_config().get_modele_path(env)
    model = load_model(model_path, compile=False)
    model.summary()

    code_path = "."
    
    images = create_dummy_db()

    sse_generator = NNCG()
    sse_generator.keras_compile(images, model, code_path, model_path.strip('.h5'), arch="general", testing=-1)

if __name__ == '__main__':
    main()
