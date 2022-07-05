import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import config as cfg
import os
import YOLO.utils


def get_dossiers(env='Simulation'):
    return cfg.get_dossier(env, 'Brut'), cfg.get_dossier(env, 'YCbCr'), cfg.get_dossier(env, 'RGB')

def brut_2_png(path_entree:str, path_sortie:str, convert_to_rgb:bool, env:str):
    dossier_entree = Path(path_entree).glob('**/batch_*')
    fichiers = [x.as_posix() for x in dossier_entree if 'label' not in str(x)]
    image_height, image_width = cfg.get_image_resolution()
    for fichier in tqdm(fichiers):
        #if '/' in fichier.split('Brut/')[1]:
        #    folder = fichier.split('Brut')[0] + 'YCbCr/' + fichier.split('Brut/')[1].split('/')[0]
        #    if not os.path.exists(folder):
        #        os.mkdir(folder)
        #new_path_sortie = path_sortie + fichier.split('Brut/')[-1] + ".png"
        repertoire_sortie = path_sortie + fichier.split('/')[-1].split('_image')[0]
        new_path_sortie = repertoire_sortie + '/' + fichier.split('/')[-1] + '.png'
        if not os.path.exists(repertoire_sortie):
            os.makedirs(repertoire_sortie)
        if not os.path.isfile(new_path_sortie):
            f = np.fromfile(fichier, dtype=np.float32)
            if env == 'Genere':
                f = np.reshape(f, (cfg.cycle_gan_image_height, cfg.cycle_gan_image_width, 3))
                f = Image.fromarray((f*255).astype(np.uint8)).resize((image_width, image_height))
                f = np.array(f) / 255.0
            else:
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
    args = YOLO.utils.parse_args_env_cam('Convertit toutes les images brutes en PNGs dans un dossier adjacent.',
                                         genere=True,
                                         hardnegative=True,
                                         testrobot=True)

    env = YOLO.utils.set_config(args, True, True)
    
    dossier_brut, dossier_YCbCr, dossier_RGB = get_dossiers(env)
    print("De " + dossier_brut + " vers " + dossier_YCbCr)
    brut_2_png(dossier_brut, dossier_YCbCr, False, env)
    print("De " + dossier_brut + " vers " + dossier_RGB)
    brut_2_png(dossier_brut, dossier_RGB, True, env)

if __name__ == '__main__':
    main()
