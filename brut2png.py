import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import config as cfg
import os.path

def brut_2_png(path_entree:str, path_sortie:str):
    dossier_entree = Path(path_entree).glob('batch_*')
    fichiers = [str(x) for x in dossier_entree if 'label' not in str(x)]
    for fichier in tqdm(fichiers):
        new_path_sortie = path_sortie + fichier.split('/')[-1] + ".png"
        if not os.path.isfile(new_path_sortie):
            f = np.fromfile(fichier, dtype=np.float32)
            f = np.reshape(f, (cfg.image_height,cfg.image_width,3))
            image = Image.fromarray((f*255).astype('uint8'))
            image.save(new_path_sortie, "png")

def main():
    print("De " + cfg.dossier_brut_simulation + " vers " + cfg.dossier_PNG_simulation)
    brut_2_png(cfg.dossier_brut_simulation, cfg.dossier_PNG_simulation)
    print("De " + cfg.dossier_brut_robot + " vers " + cfg.dossier_PNG_robot)
    brut_2_png(cfg.dossier_brut_robot, cfg.dossier_PNG_robot)

    
if __name__ == '__main__':
    main()