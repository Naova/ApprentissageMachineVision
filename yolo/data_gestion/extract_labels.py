from skimage.draw import rectangle_perimeter

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov
import yolo.utils.args_parser as args_parser

def extract_labels(path_entree:str, etiquette_path:str):
    dossier_entree = Path(path_entree).glob('*')
    fichiers = [str(x) for x in dossier_entree if x.is_file() and 'gitignore' not in str(x)]
    #match toutes les images etiquetees avec leurs etiquettes
    labels = [f for f in fichiers if 'label' in f]
    images = [l.replace('_label', '') for l in labels]

    d = {}

    image_height, image_width = cfg_prov.get_config().get_image_resolution()

    for label in tqdm(labels):
        label = label.replace('\\', '/')
        fichier = label.replace('_label', '')
        if '.gitignore' not in fichier:
            l = np.fromfile(label, dtype=np.float32)
            x = l[0]/2
            y = l[1]/2
            r = l[2]/2
            top = int(y - r)
            left = int(x - r)
            right = int(x + r)
            bottom = int(y + r)
            position = {}
            position['left'] = left
            position['right'] = right
            position['top'] = top
            position['bottom'] = bottom
            position['categorie'] = 1
            d[fichier.split('/')[-1]] = [position]
    
    #met a jour le fichier labels.json
    with open(etiquette_path) as fichier:
        labels = json.loads(fichier.read())
    for im in d:
        labels[im] = d[im]
    with open(etiquette_path, 'w') as fichier:
        json.dump(labels, fichier)

def main():
    args = args_parser.parse_args_env_cam('Convertit toutes les images brutes en PNGs dans un dossier adjacent.',
                                         genere=True,
                                         hardnegative=True,
                                         testrobot=True)

    env = args_parser.set_config(args, True, True)
    #dossier_tempo = f'{cfg_prov.get_config().naovaCodePath}/Dataset/{cfg_prov.get_config().camera}/'
    dossier_tempo = cfg_prov.get_config().get_dossier(env, 'Brut')
    labels = cfg_prov.get_config().get_labels_path('Robot')
    print(labels)
    extract_labels(dossier_tempo, labels)
    
if __name__ == '__main__':
    main()
