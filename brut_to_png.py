import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import config as cfg
import os.path

def main():
    dossier_entree = Path(cfg.dossier_brut).glob('**/*')
    dossier_sortie = cfg.dossier_PNG
    fichiers = [str(x) for x in dossier_entree if x.is_file()]

    for fichier in tqdm(fichiers):
        path_sortie = dossier_sortie + fichier.split('\\')[-1] + ".png"
        if not os.path.isfile(path_sortie) and '.gitignore' not in fichier:
            f = np.fromfile(fichier, dtype=np.float32)
            f = np.reshape(f, (cfg.image_height,cfg.image_width,3))
            image = Image.fromarray((f*255).astype('uint8'))
            image.save(path_sortie, "png")
if __name__ == '__main__':
    main()