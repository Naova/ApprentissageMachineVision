import numpy as np
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
        height, width = cfg.get_image_resolution(cfg.camera)
        if len(f) < height*width*3:
            print(fichier)
            os.remove(fichier)

def cleanup_labels(path_entree:str, dossier:str):
    with open(path_entree) as input_file:
        labels = json.loads(input_file.read())
    for image_name in labels:
        if not Path(dossier + image_name).is_file():
            breakpoint()

def main():
    dossier = cfg.get_dossier('Robot')
    print(dossier)
    cleanup_brut(dossier)
    labels = cfg.get_labels_path('Robot')
    cleanup_labels(labels, dossier)

if __name__ == '__main__':
    main()
