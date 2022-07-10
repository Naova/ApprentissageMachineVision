import tensorflow.keras as keras

import numpy as np
import os
import tqdm
import shutil
import json
import sys
sys.path.insert(0,'..')

import config as cfg
import utils
from Dataset_Loader import lire_toutes_les_images

def main():
    args = utils.parse_args_env_cam('Test the yolo model on a bunch of negatives images to find hard negatives.')
    env = utils.set_config(args)
    modele = keras.models.load_model(cfg.get_modele_path(env))
    modele.summary()

    data = lire_toutes_les_images('../'+cfg.get_dossier('NewNegative'))

    max_confidences = []

    for entree in tqdm.tqdm(data):
        entree_x = entree.x()
        prediction = modele.predict(np.array([entree_x]))[0]
        max_confidence = prediction[:,:,0].max().astype('float')
        if max_confidence > 0.2:
            max_confidences.append((entree, max_confidence))
    
    print(len(max_confidences))

    destination = '../'+cfg.get_dossier('HardNegative')

    images = []

    for entree, max_confidence in max_confidences:
        dest = destination + 'batch_' + entree.image_path.split('batch_')[-1].split('_image')[0]
        if not os.path.exists(dest):
            os.makedirs(dest, exist_ok=True)
        try:
            shutil.move(entree.image_path, entree.image_path.replace('NewNegative', 'HardNegative'))
            images.append(entree.nom)
        except:
            pass
    name = 'changes/changes.json'
    n = 1
    while os.path.exists(name):
        name = f'changes/changes{n}.json'
        n += 1
    with open(name, 'w') as f:
        json.dump(images, f)

if __name__ == '__main__':
    main()
