import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, SeparableConv2D, LeakyReLU

import matplotlib.pyplot as plt
import numpy as np
from time import process_time

from typing import Any

import sys
sys.path.insert(0,'..')

import config as cfg
from Dataset_Loader import create_dataset, lire_entrees
import utils


def kernel(x):
    return (x, x)

def create_model_upper_simulation():
    inputs = keras.Input(shape=(*cfg.get_resized_image_resolution(), 3))
    x = SeparableConv2D(16, kernel(3), kernel(2), padding='same', bias_initializer='random_normal')(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = SeparableConv2D(24, kernel(3), kernel(2), padding='same', bias_initializer='random_normal')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = SeparableConv2D(32, kernel(3), kernel(2), padding='same', bias_initializer='random_normal')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = SeparableConv2D(48, kernel(3), kernel(1), padding='same', bias_initializer='random_normal')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(32, kernel(1), kernel(1), bias_initializer='random_normal')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(5 + len(cfg.get_anchors()), kernel(1), kernel(1), activation='sigmoid', bias_initializer='random_normal')(x)
    return keras.Model(inputs=inputs, outputs=x)

def create_model_lower_simulation():
    inputs = keras.Input(shape=(*cfg.get_resized_image_resolution(), 3))
    x = SeparableConv2D(16, kernel(3), kernel(2))(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = SeparableConv2D(24, kernel(3), kernel(1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = SeparableConv2D(32, kernel(3), kernel(2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPool2D()(x)
    x = Conv2D(32, kernel(1), kernel(1))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(5 + len(cfg.get_anchors()), kernel(1), kernel(1), activation='sigmoid')(x)
    return keras.Model(inputs=inputs, outputs=x)

def create_model_upper_robot():
    inputs = keras.Input(shape=(*cfg.get_resized_image_resolution(), 3))
    x = SeparableConv2D(16, kernel(3), kernel(2), padding='same', bias_initializer='random_normal')(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(16, kernel(3), kernel(1), padding='same', bias_initializer='random_normal')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = SeparableConv2D(24, kernel(3), kernel(2), padding='same', bias_initializer='random_normal')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(24, kernel(3), kernel(1), padding='same', bias_initializer='random_normal')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = SeparableConv2D(32, kernel(3), kernel(2), padding='same', bias_initializer='random_normal')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = SeparableConv2D(32, kernel(3), kernel(1), padding='same', bias_initializer='random_normal')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(32, kernel(1), kernel(1), bias_initializer='random_normal')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(5 + len(cfg.get_anchors()), kernel(1), kernel(1), activation='sigmoid', bias_initializer='random_normal')(x)
    return keras.Model(inputs=inputs, outputs=x)

def create_model_lower_robot():
    inputs = keras.Input(shape=(*cfg.get_resized_image_resolution(), 3))
    x = SeparableConv2D(64, kernel(3), kernel(2))(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = SeparableConv2D(48, kernel(3), kernel(1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = SeparableConv2D(48, kernel(3), kernel(2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = SeparableConv2D(48, kernel(3), kernel(1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPool2D()(x)
    x = Conv2D(64, kernel(1), kernel(1))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(5 + len(cfg.get_anchors()), kernel(1), kernel(1), activation='sigmoid')(x)
    return keras.Model(inputs=inputs, outputs=x)

def create_model(env):
    if env == 'Genere':
        if cfg.camera == 'upper':
            return create_model_upper_robot()
        else:
            return create_model_lower_robot()
    else:
        if cfg.camera == 'upper':
            return create_model_upper_simulation()
        else:
            return create_model_lower_simulation()

def train_model(modele, train_generator, validation_generator):
    modele.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')
    es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10, restore_best_weights=True)
    mc = keras.callbacks.ModelCheckpoint('yolo_modele_robot_upper_{epoch:02d}.h5', monitor='val_loss')
    modele.fit(train_generator, validation_data=validation_generator, epochs=100, callbacks=[es, mc])
    return modele

def train(train_generator, validation_generator, test_data, modele_path, env, test=True):
    if cfg.retrain:
        modele = create_model(env)
        modele.summary()
        cfg.set_yolo_resolution(modele.output_shape[1], modele.output_shape[2])
        modele = train_model(modele, train_generator, validation_generator)
        modele.save(modele_path, include_optimizer=False)
        print('sauvegarde du modele : ' + modele_path)
    else:
        modele = keras.models.load_model(modele_path)
        modele.summary()
        cfg.set_yolo_resolution(modele.output_shape[1], modele.output_shape[2])
    
    if test:
        for i, entree in enumerate(test_data):
            entree_x = entree.x()
            start = process_time()
            prediction = modele.predict(np.array([entree_x]))[0]
            stop = process_time()
            print(entree.nom + ' : ', stop - start)
            utils.generate_prediction_image(prediction, entree_x, entree.y(), i)

def main():
    args = utils.parse_args_env_cam('Train a yolo model to detect balls on an image.')
    env = utils.set_config(args, use_robot=False)

    labels = cfg.get_labels_path(env)
    dossier_ycbcr = cfg.get_dossier(env, 'YCbCr')
    modele_path = cfg.get_modele_path(env)
    train_generator, validation_generator, test_data = create_dataset(16, '../'+labels, '../'+dossier_ycbcr, env)
    if not args.simulation:
        test_data = lire_entrees('../'+cfg.get_labels_path('Robot'), '../'+cfg.get_dossier('Robot'), 'Robot')
        #test_data = lire_toutes_les_images('../'+cfg.get_dossier('RobotSansBalle'))
    train(train_generator, validation_generator, test_data, modele_path, env, True)


if __name__ == '__main__':
    main()
