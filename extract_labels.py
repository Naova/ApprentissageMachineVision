from skimage.draw import rectangle_perimeter

import numpy as np
import json
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import config as cfg
import os.path
import shutil

def main():
    dossier_entree = Path(cfg.dossier_brut).glob('**/*')
    dossier_sortie = cfg.dossier_PNG
    fichiers = [str(x) for x in dossier_entree if x.is_file()]
    images = [f for f in fichiers if 'label' not in f]
    labels = [f for f in fichiers if 'label' in f]

    d = {}

    for fichier, label in tqdm(zip(images, labels)):
        path_sortie = dossier_sortie + fichier.split('\\')[-1] + ".png"
        if '.gitignore' not in fichier:
            f = np.fromfile(fichier, dtype=np.float32)
            f = np.reshape(f, (cfg.image_height,cfg.image_width,3))
            image = Image.fromarray((f*255).astype('uint8'))
            image.save(path_sortie)
            l = np.fromfile(label, dtype=np.float32)
            x = l[0]/2
            y = l[1]/2
            r = l[2]/2
            top = int(y - r)
            left = int(x - r)
            right = int(x + r)
            bottom = int(y + r)
            d[fichier] = []
            position = {}
            position['left'] = left
            position['right'] = right
            position['top'] = top
            position['bottom'] = bottom
            position['categorie'] = 1
            d[fichier].append(position)
            #rect = rectangle_perimeter((top, left), (bottom, right), shape=(cfg.image_height, cfg.image_width), clip=True)
            #f[rect] = 1
            
            #image = Image.fromarray((f*255).astype('uint8'))
            #image.show()
    with open(cfg.json_etiquettes) as fichier:
        labels = json.loads(fichier.read())
    for im in d:
        labels[im] = d[im]
    with open(cfg.json_etiquettes, 'w') as fichier:
        json.dump(labels, fichier)
if __name__ == '__main__':
    main()