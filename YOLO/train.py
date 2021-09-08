from Dataset_Loader import create_dataset

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, LeakyReLU, SeparableConv2D
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import random
import numpy as np

from time import process_time
import utils

import sys
sys.path.insert(0,'..')
import config as cfg

def kernel(x):
    return (x, x)

def add_conv_2d(x, n_filters=16, kernel=kernel(3), stride=kernel(1), ConvType=Conv2D):
    x = ConvType(n_filters, kernel, stride)(x)
    x = LeakyReLU()(x)
    return x

def create_model(shape:tuple, nb_anchors:int):
    inputs = keras.layers.Input(shape=shape)
    x = add_conv_2d(inputs, 64, kernel(5), kernel(2), Conv2D)
    
    x = add_conv_2d(x, 64, kernel(5), kernel(1), SeparableConv2D)
    x = add_conv_2d(x, 64, kernel(3), kernel(1), SeparableConv2D)
    x = add_conv_2d(x, 64, kernel(3), kernel(1), SeparableConv2D)
    x = MaxPool2D(kernel(2))(x)

    x = Dense(64)(x)
    x = LeakyReLU()(x)
    x = Conv2D(3 + nb_anchors, kernel(1), activation='sigmoid')(x)
    return keras.models.Model(inputs, x)

def train_model(modele, train_generator, validation_generator):
    modele.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')
    es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, restore_best_weights=True)
    modele.fit(train_generator, validation_data=validation_generator, epochs=40, callbacks=[es])
    return modele

def display_model_prediction(prediction, wanted_prediction, prediction_on_image, wanted_output, save_to_file_name = None):
    fig = plt.figure()
    fig.add_subplot(2, 2, 1)
    plt.imshow(prediction)
    plt.title('model output')
    plt.colorbar(orientation='horizontal')
    fig.add_subplot(2, 2, 2)
    plt.imshow(wanted_prediction)
    plt.title('ground truth')
    plt.colorbar(orientation='horizontal')
    fig.add_subplot(2, 2, 3)
    plt.imshow(prediction_on_image)
    plt.title('model output on image')
    fig.add_subplot(2, 2, 4)
    plt.imshow(wanted_output)
    plt.title('ground truth on image')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    if save_to_file_name:
        plt.savefig('predictions/' + save_to_file_name, dpi=300)
    plt.show()

def generate_prediction_image(prediction, x_test, y_test, prediction_number = None):
    coords = utils.n_max_coord(prediction[:,:,0], 1)
    prediction_on_image = utils.draw_rectangle_on_image(utils.ycbcr2rgb(x_test.copy()), prediction, coords)
    coords = utils.treshold_coord(y_test[:,:,0])
    wanted_output = utils.draw_rectangle_on_image(utils.ycbcr2rgb(x_test.copy()), y_test, coords)
    display_model_prediction(prediction[:,:,0], y_test[:,:,0], prediction_on_image, wanted_output, 'prediction_' + str(prediction_number) + '.png')

def train(train_generator, validation_generator, test_generator, modele_path, test=True):
    shape = (cfg.resized_image_height, cfg.resized_image_width, 3)
    if cfg.retrain:
        modele = create_model(shape, cfg.yolo_nb_anchors)
        modele.summary()
        modele = train_model(modele, train_generator, validation_generator)
        modele.save(modele_path, include_optimizer=False)
        print('sauvegarde du modele : ' + modele_path)
    else:
        modele = keras.models.load_model(modele_path)
    
    if test:
        for i, entree in enumerate(test_generator):
            entree_x = entree.x()
            start = process_time()
            prediction = modele.predict(np.array([entree_x]))[0]
            stop = process_time()
            print(entree.nom + ' : ', stop - start)
            generate_prediction_image(prediction, entree_x, entree.y(), i)

if __name__ == '__main__':
    #simulation
    #train_generator, validation_generator, test_generator = create_dataset(16, '../'+cfg.labels_simulation, '../'+cfg.dossier_brut_simulation)
    #train(train_generator, validation_generator, test_generator, cfg.model_path_simulation, True)
    #images generees
    train_generator, validation_generator, test_generator = create_dataset(16, '../'+cfg.labels_simulation, '../'+cfg.dossier_brut_genere)
    train(train_generator, validation_generator, test_generator, cfg.model_path_robot, True)
