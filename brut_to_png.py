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
        if not os.path.isfile(path_sortie):
            f = np.fromfile(fichier, dtype=np.int32)
            f = np.reshape(f, (320,240,3))
            image = Image.fromarray(f.astype('uint8'))
            image.save(path_sortie, "png")

    #version en une ligne parce que je m'ennuie :
    #[Image.fromarray(np.reshape(np.fromfile(str(x), dtype=np.int32), (320,240,3)).astype('uint8')).save(cfg.dossier_PNG + str(x).split('\\')[-1] + ".png", "png") for x in Path(cfg.dossier_brut).glob('**/*') if x.is_file()]

if __name__ == '__main__':
    main()