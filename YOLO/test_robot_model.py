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

def centile(x:List[float], y:float):
    return x[int(y * len(x))]

def seuil(x:List[float], y:float):
    return len([_x for _x in x if _x[2] > y])

def save_stats_as_json(y, time):
    somme = sum(y)
    stats = {
        'somme':somme,
        '0_centile':centile(y, 0),
        '25_centile':centile(y, 0.25),
        '50_centile':centile(y, 0.5),
        '75_centile':centile(y, 0.75),
        '90_centile':centile(y, 0.9),
        '95_centile':centile(y, 0.95),
        '100_centile':y[len(y)-1],
        '5_seuil':seuil(y, 0.05),
        '10_seuil':seuil(y, 0.1),
        '15_seuil':seuil(y, 0.15),
        '20_seuil':seuil(y, 0.2),
        '40_seuil':seuil(y, 0.4),
        '60_seuil':seuil(y, 0.6),
        '80_seuil':seuil(y, 0.8),
    }
    
    with open(f'tests/{cfg.camera}/{time}_raw_data.json', 'w') as f:
        json.dump(y, f)
    with open(f'tests/{cfg.camera}/{time}_stats.json', 'w') as f:
        json.dump(stats, f)

def test_model(modele, test_data):
    max_confidences = []

    for entree in tqdm.tqdm(test_data):
        entree_x = entree.x()
        prediction = modele.predict(np.array([entree_x]))[0]
        max_confidences.append((entree.nom, entree.flipper, prediction[:,:,0].max().astype('float')))
    max_confidences.sort(key=lambda x: x[2])
    return max_confidences

def save_stats(confidences_negative, confidences_positive, env):
    time = datetime.now().strftime('%Y_%m_%d-%H-%M-%S')

    y_neg = [x[2] for x in confidences_negative]
    y_pos = [x[2] for x in confidences_positive]

    treshold = 0.18

    false_negative = [x for x in y_pos if x <= treshold]
    true_negative = [x for x in y_neg if x <= treshold]
    false_positive = [x for x in y_neg if x > treshold]
    true_positive = [x for x in y_pos if x > treshold]

    #plt.scatter(range(len(confidences_negative)), y_neg, s=10)
    #plt.scatter(range(len(confidences_positive)), y_pos, s=10)
    plt.scatter(range(len(false_negative)), false_negative, s=10, color='orange')
    plt.scatter(range(len(false_negative), len(true_positive)+len(false_negative)), true_positive, s=10, color='blue')
    plt.scatter(range(len(true_negative)), true_negative, s=10, color='green')
    plt.scatter(range(len(true_negative), len(false_positive)+len(true_negative)), false_positive, s=10, color='red')
    plt.axhline(y = treshold, color = 'b', linestyle = ':')
    plt.rcParams["axes.titlesize"] = 10
    plt.title(f'Maximal confidence level per image from the test dataset.\n{cfg.camera.capitalize()} Camera')
    plt.xlabel('Images')
    plt.ylabel('Niveau de confiance')
    #plt.title(f'Maximal confidence level per image from the test dataset.\n{cfg.camera.capitalize()} camera.')
    #plt.xlabel('Images (sorted)')
    #plt.ylabel('Max confidence level')
    plt.ylim(-0.05, 1.)
    plt.legend(['Detection treshold', 'False negatives', 'True positives', 'True negatives', 'False positives'])
    plt.grid()

    plt.savefig(f'tests/{cfg.camera}/{time}.png')

    somme = sum(y_neg)
    print(somme)
    somme = sum(y_pos)
    print(somme)
    
    source = cfg.get_modele_path(env)
    destination = f'tests/{cfg.camera}/{time}.h5'
    shutil.copy(source, destination)

def main():
    args = utils.parse_args_env_cam('Test the yolo model on a bunch of test images and output stats.')
    env = utils.set_config(args)
    modele = keras.models.load_model(cfg.get_modele_path(env))
    modele.summary()
    test_data_negative = lire_toutes_les_images('../'+cfg.get_dossier('TestRobot'))
    test_data_positive = lire_entrees('../'+cfg.get_labels_path('Robot'), '../'+cfg.get_dossier('Robot'), env='Robot')

    max_confidences_negative = test_model(modele, test_data_negative)
    max_confidences_positive = test_model(modele, test_data_positive)

    save_stats(max_confidences_negative, max_confidences_positive, env)

    
if __name__ == '__main__':
    main()
