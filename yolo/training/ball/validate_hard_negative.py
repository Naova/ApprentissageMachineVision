import numpy as np
import os
import tqdm
import shutil

from yolo.utils.configuration_provider import ConfigurationProvider as cfg_prov
import yolo.utils.args_parser as args_parser
from yolo.training.dataset_loader import lire_toutes_les_images

from yolo.training.ball.train import load_model

def main():
    args = args_parser.parse_args_env_cam('Test the yolo model on a bunch of negatives images to find hard negatives.')
    env = args_parser.set_config(args)
    modele = load_model(env=env)
    modele.summary()

    data = lire_toutes_les_images(cfg_prov.get_config().get_dossier('NewNegative'))

    max_confidences = []

    for entree in tqdm.tqdm(data):
        entree_x = entree.x()
        prediction = modele.predict(np.array([entree_x]))[0]
        max_confidence = prediction[:,:,0].max().astype('float')
        if max_confidence > 0.2:
            max_confidences.append((entree, max_confidence))
    
    print(len(max_confidences))

    destination = cfg_prov.get_config().get_dossier('HardNegative')

    images = []

    for entree, max_confidence in max_confidences:
        dest = destination + 'batch_' + entree.image_path.split('batch_')[-1].split('_image')[0]
        if not os.path.exists(dest):
            os.makedirs(dest, exist_ok=True)
        shutil.move(entree.image_path, entree.image_path.replace('NewNegative', 'HardNegative'))
        images.append(entree.nom)

if __name__ == '__main__':
    main()
