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
from Dataset_Loader import lire_toutes_les_images

def centile(x:List[float], y:float):
    return x[int(y * len(x))]

def seuil(x:List[float], y:float):
    return len([_x for _x in x if _x[2] > y])

def main():
    args = utils.parse_args_env_cam('Test the yolo model on a bunch of test images and output stats.')
    env = utils.set_config(args)
    modele = keras.models.load_model(cfg.get_modele_path(env))
    modele.summary()
    test_data = lire_toutes_les_images('../'+cfg.get_dossier('RobotSansBalle'))

    max_confidences = []

    for entree in tqdm.tqdm(test_data):
        entree_x = entree.x()
        prediction = modele.predict(np.array([entree_x]))[0]
        max_confidences.append((entree.nom, entree.flipper, prediction[:,:,0].max().astype('float')))
    max_confidences.sort(key=lambda x: x[2])
    y = [x[2] for x in max_confidences]

    time = datetime.now().strftime('%Y_%m_%d-%H-%M-%S')

    plt.scatter(range(len(max_confidences)), y)
    plt.rcParams["axes.titlesize"] = 10
    plt.title(f'Niveau de confiance maximal par image du dataset de test.\nLe plus bas est le mieux.\n{time}')
    plt.xlabel('Images')
    plt.ylabel('Niveau de confiance')
    plt.ylim(-0.05, 1.)
    plt.grid()

    plt.savefig(f'tests/{time}.png')

    somme = sum([x[2] for x in max_confidences])

    print(somme)

    stats = {
        'somme':somme,
        '0_centile':centile(max_confidences, 0),
        '25_centile':centile(max_confidences, 0.25),
        '50_centile':centile(max_confidences, 0.5),
        '75_centile':centile(max_confidences, 0.75),
        '90_centile':centile(max_confidences, 0.9),
        '95_centile':centile(max_confidences, 0.95),
        '100_centile':max_confidences[len(max_confidences)-1],
        '5_seuil':seuil(max_confidences, 0.05),
        '10_seuil':seuil(max_confidences, 0.1),
        '15_seuil':seuil(max_confidences, 0.15),
        '20_seuil':seuil(max_confidences, 0.2),
        '40_seuil':seuil(max_confidences, 0.4),
        '60_seuil':seuil(max_confidences, 0.6),
        '80_seuil':seuil(max_confidences, 0.8),
    }
    
    with open(f'tests/{time}_raw_data.json', 'w') as f:
        json.dump(max_confidences, f)
    with open(f'tests/{time}_stats.json', 'w') as f:
        json.dump(stats, f)
    
    source = cfg.get_modele_path(env)
    destination = f'tests/{time}.h5'
    shutil.copy(source, destination)

    
if __name__ == '__main__':
    main()
