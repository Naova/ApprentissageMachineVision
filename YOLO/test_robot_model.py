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
    return len([_x for _x in x if _x[1] > y])

def main():
    args = utils.parse_args_env_cam('Test the yolo model on a bunch of test images and output stats.')
    env = utils.set_config(args)
    modele = keras.models.load_model(cfg.get_modele_path(env))
    modele.summary()
    test_data = lire_toutes_les_images('../'+cfg.get_dossier('RobotSansBalle'))

    max_confidences = []

    for i, entree in enumerate(tqdm.tqdm(test_data)):
        entree_x = entree.x()
        prediction = modele.predict(np.array([entree_x]))[0]
        max_confidences.append((entree.nom, prediction[:,:,0].max().astype('float')))
    max_confidences.sort(key=lambda x: x[1])
    y = [x[1] for x in max_confidences]

    time = datetime.now().strftime('%d_%m_%Y-%H-%M-%S')

    plt.scatter(range(len(max_confidences)), y)
    plt.savefig(f'tests/{time}.png')

    stats = {
        'max_confidences':max_confidences,
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
    
    with open(f'tests/{time}.json', 'w') as f:
        json.dump(stats, f)
    
    source = cfg.get_modele_path(env)
    destination = f'tests/{time}.h5'
    shutil.copy(source, destination)

    
if __name__ == '__main__':
    main()
