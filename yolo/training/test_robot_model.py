import tensorflow.keras as keras

import numpy as np
import json
import tqdm
import matplotlib.pyplot as plt
import shutil
from datetime import datetime

import yolo.config as cfg
import yolo.utils.args_parser as args_parser
from yolo.training.dataset_loader import lire_toutes_les_images, lire_entrees


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

    treshold = 0.8

    false_negative = [x for x in y_pos if x <= treshold]
    true_negative = [x for x in y_neg if x <= treshold]
    false_positive = [x for x in y_neg if x > treshold]
    true_positive = [x for x in y_pos if x > treshold]

    fp = 100 * len(false_positive) / (len(true_negative) + len(false_positive))
    fn = 100 * len(true_positive) / (len(false_negative) + len(true_positive))
    print(f'False positive : {fp}%')
    print(f'True positive : {fn}%')

    with open('stats.json', 'r') as f:
        stats = json.load(f)
    stats[time] = {
        'False positive':fp,
        'True positive' :fn,
        'y_neg':y_neg,
        'y_pos':y_pos,
    }
    with open('stats.json', 'w') as f:
        json.dump(stats, f)

    plt.scatter(range(len(false_negative)), false_negative, s=10, color='orange')
    plt.scatter(range(len(false_negative), len(true_positive)+len(false_negative)), true_positive, s=10, color='blue')
    plt.scatter(range(len(true_negative)), true_negative, s=10, color='green')
    plt.scatter(range(len(true_negative), len(false_positive)+len(true_negative)), false_positive, s=10, color='red')
    
    plt.text(0,-0.05,f'False positive : {fp}%')
    plt.text(0,0.5,f'True positive : {fn}%')

    plt.axhline(y = treshold, color = 'b', linestyle = ':')
    plt.rcParams["axes.titlesize"] = 10
    plt.title(f'Maximal confidence level per image from the test dataset.\n{cfg_prov.get_config().camera.capitalize()} camera.')
    plt.xlabel('Images (sorted)')
    plt.ylabel('Max confidence level')
    plt.ylim(-0.05, 1.)
    plt.legend(['False negatives', 'True positives', 'True negatives', 'False positives', 'Detection treshold'])
    plt.grid()

    plt.savefig(f'tests/{cfg_prov.get_config().camera}/{time}.png')

    plt.clf()

    somme_neg = sum(y_neg)
    print(somme_neg)
    somme_pos = sum(y_pos)
    print(somme_pos)
    
    source = cfg_prov.get_config().get_modele_path(env)
    destination = f'tests/{cfg_prov.get_config().camera}/{time}.h5'
    shutil.copy(source, destination)

    return somme_neg, somme_pos

def main():
    args = args_parser.parse_args_env_cam('Test the yolo model on a bunch of test images and output stats.')
    env = args_parser.set_config(args)
    modele = keras.models.load_model(cfg_prov.get_config().get_modele_path(env))
    modele.summary()
    test_data_positive = lire_toutes_les_images(cfg_prov.get_config().get_dossier('TestRobotPositive'))
    test_data_negative = lire_toutes_les_images(cfg_prov.get_config().get_dossier('TestRobot'))
    test_data_positive += lire_entrees(cfg_prov.get_config().get_labels_path('Robot'), cfg_prov.get_config().get_dossier('Robot'), env='Robot')

    max_confidences_negative = test_model(modele, test_data_negative)
    max_confidences_positive = test_model(modele, test_data_positive)

    return save_stats(max_confidences_negative, max_confidences_positive, env)

    
if __name__ == '__main__':
    main()
