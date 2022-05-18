import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, SeparableConv2D, LeakyReLU

import matplotlib.pyplot as plt
import numpy as np
from time import process_time

import sys
sys.path.insert(0,'..')

import config as cfg
from Dataset_Loader import create_dataset, lire_entrees, lire_toutes_les_images
import utils


def kernel(x):
    return (x, x)

def create_model_upper(shape:tuple, nb_anchors:int):
    inputs = keras.Input(shape=(*cfg.get_resized_image_resolution(), 3))
    x = SeparableConv2D(48, kernel(3), kernel(2))(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = SeparableConv2D(48, kernel(3), kernel(2))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPool2D()(x)
    x = Conv2D(32, kernel(1), kernel(1))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(3 + nb_anchors, kernel(1), kernel(1), activation='sigmoid')(x)
    return keras.Model(inputs=inputs, outputs=x)

def create_model_lower(shape:tuple, nb_anchors:int):
    inputs = keras.Input(shape=(*cfg.get_resized_image_resolution(), 3))
    x = SeparableConv2D(32, kernel(5), kernel(2))(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = SeparableConv2D(32, kernel(3), kernel(2))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPool2D()(x)
    x = Conv2D(16, kernel(1), kernel(1))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(3 + nb_anchors, kernel(1), kernel(1), activation='sigmoid')(x)
    return keras.Model(inputs=inputs, outputs=x)

def create_model(shape:tuple, nb_anchors:int):
    if cfg.camera == 'upper':
        return create_model_upper(shape, nb_anchors)
    else:
        return create_model_lower(shape, nb_anchors)

def train_model(modele, train_generator, validation_generator):
    modele.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')
    es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=2, restore_best_weights=True)
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
    if save_to_file_name:
        plt.savefig('predictions/' + save_to_file_name, dpi=300)
    plt.show()

def generate_prediction_image(prediction, x_test, y_test, prediction_number = None):
    coords = utils.n_max_coord(prediction[:,:,0], 1)
    #coords = utils.non_max_suppression(prediction)
    prediction_on_image = utils.draw_rectangle_on_image(utils.ycbcr2rgb(x_test.copy()), prediction, coords)
    coords = utils.treshold_coord(y_test[:,:,0])
    wanted_output = utils.draw_rectangle_on_image(utils.ycbcr2rgb(x_test.copy()), y_test, coords)
    display_model_prediction(prediction[:,:,0], y_test[:,:,0], prediction_on_image, wanted_output, 'prediction_' + str(prediction_number) + '.png')

def train(train_generator, validation_generator, test_data, modele_path, test=True):
    resized_image_height, resized_image_width = cfg.get_resized_image_resolution()
    shape = (resized_image_height, resized_image_width, 3)
    if cfg.retrain:
        modele = create_model(shape, cfg.get_nb_anchors())
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
            generate_prediction_image(prediction, entree_x, entree.y(), i)

def main():
    args = utils.parse_args_env_cam('Train a yolo model to detect balls on an image.')
    env = utils.set_config(args, use_robot=False)

    labels = cfg.get_labels_path(env)
    dossier_brut = cfg.get_dossier(env, 'Brut')
    modele_path = cfg.get_modele_path(env)
    train_generator, validation_generator, test_data = create_dataset(16, '../'+labels, '../'+dossier_brut, env)
    if not args.simulation:
        #test_data = lire_entrees('../'+cfg.get_labels_path('Robot'), '../'+cfg.get_dossier('Robot'), 'Robot')
        test_data = lire_toutes_les_images('../'+cfg.get_dossier('RobotSansBalle'))
    train(train_generator, validation_generator, test_data, modele_path, True)


if __name__ == '__main__':
    main()
