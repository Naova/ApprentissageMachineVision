#!/usr/bin/env python3
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras
from nncg.nncg import NNCG

import yolo.utils.args_parser as args_parser
from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov
from yolo.training.ball.train import custom_activation


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def create_dummy_db():
    return [np.zeros((120, 160, 3))]

def main():
    args = args_parser.parse_args_env_cam('Convertit un modele keras (.h5) en un fichier .cpp reproduisant l\'execution du modele.')
    env = args_parser.set_config(args)
    model_path = cfg_prov.get_config().get_modele_path(env)
    model = keras.models.load_model(model_path, custom_objects={'custom_activation':custom_activation}, compile=False)
    model.summary()

    #pas particulierement catholique
    if model.layers[-1].activation.__name__ == "custom_activation":
        new_last_layer = keras.layers.Conv2D(5 + len(cfg_prov.get_config().get_anchors()), (1, 1), (1, 1), activation='sigmoid')(model.layers[-2].output)
        model2 = keras.Model(inputs=model.layers[0].input, outputs=new_last_layer)
        model2.set_weights(model.get_weights())
        model = model2

    code_path = "."
    
    images = create_dummy_db()

    sse_generator = NNCG()
    try:
        sse_generator.keras_compile(images, model, code_path, model_path.strip('.h5'), arch="general", testing=-1)
    except Exception as err:
        print(Exception, err)
        print("Please ignore the previous exception. The generated code file is probably there.")

if __name__ == '__main__':
    main()
