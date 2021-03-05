from skimage.draw import rectangle_perimeter

import numpy as np
import json
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import config as cfg
import os.path
import shutil

def extract_labels(path_entree:str, path_sortie:str, etiquette_path:str):
    dossier_entree = Path(path_entree).glob('**/*')
    fichiers = [str(x) for x in dossier_entree if x.is_file() and 'gitignore' not in str(x)]
    images = [f for f in fichiers if 'label' not in f]
    labels = [f for f in fichiers if 'label' in f]

    images.sort()
    labels.sort()

    d = {}

    for fichier, label in tqdm(zip(images, labels)):
        fichier = fichier.replace('\\', '/')
        label = label.replace('\\', '/')
        if '.gitignore' not in fichier:
            f = np.fromfile(fichier, dtype=np.float32)
            f = np.reshape(f, (cfg.image_height,cfg.image_width,3))
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
    with open(etiquette_path) as fichier:
        labels = json.loads(fichier.read())
    for im in d:
        labels[im] = d[im]
    with open(etiquette_path, 'w') as fichier:
        json.dump(labels, fichier)
    for i in images:
        i = i.replace('\\', '/')
        shutil.copy(i, path_sortie + i.split('/')[-1])

def main():
    extract_labels(cfg.dossier_tempo_simulation, cfg.dossier_brut_simulation, cfg.labels_simulation)
    extract_labels(cfg.dossier_tempo_robot, cfg.dossier_brut_robot, cfg.labels_robot)
    
if __name__ == '__main__':
    main()
