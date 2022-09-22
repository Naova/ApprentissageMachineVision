from skimage.draw import rectangle_perimeter

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import yolo.training.ball.config as cfg
import yolo.config as cfg_global
import os.path
import os
import shutil

def extract_labels(path_entree:str, path_sortie:str, etiquette_path:str):
    dossier_entree = Path(path_entree).glob('**/*')
    fichiers = [str(x) for x in dossier_entree if x.is_file() and 'gitignore' not in str(x)]

    #match toutes les images etiquetees avec leurs etiquettes
    labels = [f for f in fichiers if 'label' in f]
    images = [l.replace('_label', '') for l in labels]

    d = {}

    image_height, image_width = cfg.get_image_resolution()

    for fichier, label in tqdm(zip(images, labels)):
        fichier = fichier.replace('\\', '/')
        label = label.replace('\\', '/')
        if '.gitignore' not in fichier:
            f = np.fromfile(fichier, dtype=np.float32)
            if len(f) == 0: #les fichiers provenant du robot sont parfois vides par manque d'espace disque
                print(fichier)
                os.remove(fichier)
                os.remove(fichier + '_label')
                continue
            try:
                f = np.reshape(f, (image_height, image_width, 3))
            except:
                print(fichier)
                breakpoint()
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
    
    #deplace toutes les images (balle ou pas) dans le dossier de destination
    for i in images:
        i = i.replace('\\', '/')
        shutil.copy(i, path_sortie + i.split('/')[-1])

def main():
    dossier_tempo = f'{cfg_global.naovaCodePath}/Dataset/{cfg.camera}/'
    dossier_brut = cfg.get_dossier('Robot', 'Brut')
    print(dossier_brut)
    labels = cfg.get_labels_path('Robot')
    print(labels)
    extract_labels(dossier_tempo, dossier_brut, labels)
    
if __name__ == '__main__':
    main()
