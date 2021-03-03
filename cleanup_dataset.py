import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import config as cfg
import os.path
import os
import json

def cleanup_brut(path_entree:str):
    dossier_entree = Path(path_entree).glob('batch_*')
    fichiers = [str(x) for x in dossier_entree if 'label' not in str(x)]
    for fichier in tqdm(fichiers):
        f = np.fromfile(fichier, dtype=np.float32)
        if len(f) < cfg.image_height*cfg.image_width*3:
            print(fichier)
            #breakpoint()
            os.remove(fichier)

def cleanup_labels(path_entree:str):
    with open(path_entree) as input_file:
        labels = json.loads(input_file.read())
    for image_name in labels:
        if not Path(cfg.dossier_brut_simulation + image_name).is_file():
            breakpoint()

def main():
    print(cfg.dossier_brut_robot)
    cleanup_brut(cfg.dossier_brut_robot)
    cleanup_labels(cfg.labels_simulation)

if __name__ == '__main__':
    main()
