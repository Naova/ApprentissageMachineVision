import train
from Dataset_Loader import lire_entrees
import utils

import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np
import tqdm

import sys
sys.path.insert(0,'..')
import config as cfg

def display_model_prediction(prediction_simu, prediction_robot, wanted_prediction, prediction_on_image_simu, prediction_on_image_robot, wanted_output, save_to_file_name = None):
    fig = plt.figure()
    
    fig.add_subplot(2, 3, 1, title='simulation output')
    plt.imshow(prediction_simu[:,:,0])
    plt.colorbar(orientation='horizontal')
    fig.add_subplot(2, 3, 4, title='simulation output on image')
    plt.imshow(prediction_on_image_simu)

    fig.add_subplot(2, 3, 2, title='robot output')
    plt.imshow(prediction_robot[:,:,0])
    plt.colorbar(orientation='horizontal')
    fig.add_subplot(2, 3, 5, title='robot output on image')
    plt.imshow(prediction_on_image_robot)

    fig.add_subplot(2, 3, 3, title='ground truth')
    plt.imshow(wanted_prediction)
    plt.colorbar(orientation='horizontal')
    fig.add_subplot(2, 3, 6, title='ground truth on image')
    plt.imshow(wanted_output)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    #if save_to_file_name:
    #    plt.savefig('predictions/' + save_to_file_name, dpi=300)
    plt.show()

def generate_predictions_on_image(prediction, x_test, y_test, prediction_number = None):
    coords = utils.n_max_coord(prediction[:,:,0], 1)
    prediction_on_image = utils.draw_rectangle_on_image(utils.ycbcr2rgb(x_test.copy()), prediction, coords)
    coords = utils.treshold_coord(y_test[:,:,0])
    wanted_output = utils.draw_rectangle_on_image(utils.ycbcr2rgb(x_test.copy()), y_test, coords)
    return prediction_on_image, wanted_output

def test_model_on_dataset(modele_simulation, modele_robot, dataset):
    losses_simulation = []
    losses_robot = []
    print(len(dataset))
    for i, entree in tqdm.tqdm(enumerate(dataset)):
        entree_x = entree.x()
        entree_y = entree.y()
        losses_simulation.append(modele_simulation.evaluate(np.array([entree_x]), np.array([entree_y]), verbose=0))
        prediction_simulation = modele_simulation.predict(np.array([entree_x]))[0]
        losses_robot.append(modele_robot.evaluate(np.array([entree_x]), np.array([entree_y]), verbose=0))
        print('\n')
        print(losses_simulation)
        print(losses_robot)
        prediction_robot = modele_robot.predict(np.array([entree_x]))[0]
        prediction_on_image_simu, wanted_output = generate_predictions_on_image(prediction_simulation, entree_x, entree_y, i)
        prediction_on_image_robot, wanted_output = generate_predictions_on_image(prediction_robot, entree_x, entree_y, i)
        display_model_prediction(prediction_simulation, prediction_robot, entree_y[:,:,0], prediction_on_image_simu, prediction_on_image_robot, wanted_output, f'prediction{i}.png')
    print('total simulation : ' + str(sum(losses_simulation)))
    print('moyenne simulation : ' + str(sum(losses_simulation) / len(losses_simulation)))
    print('total robot : ' + str(sum(losses_robot)))
    print('moyenne robot : ' + str(sum(losses_robot) / len(losses_robot)))

def main(dataset_robot):
    modele_simulation = keras.models.load_model(cfg.model_path_simulation)
    modele_simulation.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')
    modele_robot = keras.models.load_model(cfg.model_path_robot)
    modele_robot.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')

    test_model_on_dataset(modele_simulation, modele_robot, dataset_robot)

if __name__ == '__main__':
    dataset_robot = lire_entrees('../'+cfg.labels_robot, '../'+cfg.dossier_brut_robot)
    main(dataset_robot)