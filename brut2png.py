import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import config as cfg
import os.path


def get_dossiers(env='Simulation'):
    return cfg.get_dossier(env, 'Brut'), cfg.get_dossier(env, 'PNG')

def brut_2_png(path_entree:str, path_sortie:str, convert_to_rgb:bool):
    dossier_entree = Path(path_entree).glob('batch_*')
    fichiers = [str(x) for x in dossier_entree if 'label' not in str(x)]
    image_height, image_width = cfg.get_image_resolution()
    for fichier in tqdm(fichiers):
        fichier = fichier.replace('\\', '/')
        new_path_sortie = path_sortie + fichier.split('/')[-1] + ".png"
        if not os.path.isfile(new_path_sortie):
            f = np.fromfile(fichier, dtype=np.float32)
            f = np.reshape(f, (image_height, image_width, 3))
            arr = f*255
            if convert_to_rgb:
                xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
                arr[:,:,[1,2]] -= 128
                arr = arr.dot(xform.T)
                np.putmask(arr, arr > 255, 255)
                np.putmask(arr, arr < 0, 0)
            image = Image.fromarray((arr).astype('uint8'))
            image.save(new_path_sortie, "png")

def main():
    dossier_brut, dossier_PNG = get_dossiers('Simulation')
    print("De " + dossier_brut + " vers " + dossier_PNG)
    brut_2_png(dossier_brut, dossier_PNG, True)

if __name__ == '__main__':
    main()
