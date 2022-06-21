import tensorflow.keras as keras

from typing import List
import numpy as np
import json
import tqdm
import matplotlib.pyplot as plt
import shutil
from datetime import datetime
import sys
sys.path.insert(0,'..')

import config as cfg
import utils
from Dataset_Loader import lire_toutes_les_images, lire_entrees
from test_robot_model import test_model

"""
Copie honteusement d'internet
"""
def gradient_descent(y, gradient, start, learn_rate, n_iter):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(y, vector)
        vector += diff
    return vector

#def fonction_gradiant(y:List[float], x:float):

def test_datapoints(y_neg, y_pos, negative_weight=20):
    pourcentage_pos = []
    pourcentage_neg = []
    combined = []
    nb_points = 500
    for i in range(0, nb_points):
        pos = len([y for y in y_pos if y > i / (nb_points*2)]) / len(y_pos)
        neg = len([y for y in y_neg if y > i / (nb_points*2)]) / len(y_neg)
        pourcentage_pos.append(pos)
        pourcentage_neg.append(neg)
        combined.append(pos - (neg * negative_weight))
    print(combined.index(max(combined))/(nb_points*2))
    plt.scatter([0.5*(i/nb_points) for i in range(len(pourcentage_neg))], pourcentage_neg, s=10)
    plt.scatter([0.5*(i/nb_points) for i in range(len(pourcentage_pos))], pourcentage_pos, s=10)
    plt.scatter([0.5*(i/nb_points) for i in range(len(combined))], combined, s=10)
    plt.ylim(-0.5, 1.)
    plt.legend(['Pourcentage de faux positifs acceptés', 'Pourcentage de vrai positifs acceptés', 'Score global'])
    plt.xlabel('Seuil de confiance')
    plt.grid()
    plt.show()
    #plt.savefig(f'tests/{cfg.camera}/{time}.png')


def main():
    args = utils.parse_args_env_cam('Test the yolo model on a bunch of test images and output stats.')
    env = utils.set_config(args)
    modele = keras.models.load_model(cfg.get_modele_path(env))
    modele.summary()
    test_data_negative = lire_toutes_les_images('../'+cfg.get_dossier('TestRobot'))
    test_data_positive = lire_entrees('../'+cfg.get_labels_path('Robot'), '../'+cfg.get_dossier('Robot'), env='Robot')
    
    max_confidences_negative = test_model(modele, test_data_negative)
    max_confidences_positive = test_model(modele, test_data_positive)

    y_neg = [x[2] for x in max_confidences_negative]
    y_pos = [x[2] for x in max_confidences_positive]

    test_datapoints(y_neg, y_pos)


if __name__ == '__main__':
    main()
