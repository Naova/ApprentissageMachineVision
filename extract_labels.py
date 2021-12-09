from skimage.draw import rectangle_perimeter

import numpy as np
import json
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

    image_height, image_width = cfg.get_image_resolution()

    for fichier, label in tqdm(zip(images, labels)):
        fichier = fichier.replace('\\', '/')
        label = label.replace('\\', '/')
        if '.gitignore' not in fichier:
            f = np.fromfile(fichier, dtype=np.float32)
            f = np.reshape(f, (image_height, image_width, 3))
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
    dossier_tempo = f'../NaovaCode/Dataset/{cfg.camera}/'
    dossier_brut = cfg.get_dossier('Simulation', 'Brut')
    labels = cfg.get_labels_path('Simulation')
    extract_labels(dossier_tempo, dossier_brut, labels)
    
if __name__ == '__main__':
    main()
