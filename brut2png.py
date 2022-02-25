import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import config as cfg
import os
import argparse


def get_dossiers(env='Simulation'):
    return cfg.get_dossier(env, 'Brut'), cfg.get_dossier(env, 'PNG')

def brut_2_png(path_entree:str, path_sortie:str, convert_to_rgb:bool, env:str):
    dossier_entree = Path(path_entree).glob('**/batch_*')
    fichiers = [str(x) for x in dossier_entree if 'label' not in str(x)]
    image_height, image_width = cfg.get_image_resolution()
    for fichier in tqdm(fichiers):
        if '/' in fichier.split('Brut/')[1]:
            folder = fichier.split('Brut')[0] + 'PNG/' + fichier.split('Brut/')[1].split('/')[0]
            if not os.path.exists(folder):
                os.mkdir(folder)
        new_path_sortie = path_sortie + fichier.split('Brut/')[-1] + ".png"
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
    parser = argparse.ArgumentParser(description='Convertit toutes les images brutes en PNGs dans un dossier adjacent.')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('-s', '--simulation', action='store_true',
                        help='Traiter les images de simulation.')
    action.add_argument('-r', '--robot', action='store_true',
                        help='Traiter les photos du robot.')
    action.add_argument('-g', '--genere', action='store_true',
                        help='Traiter les images generees par CycleGAN.')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('-u', '--upper', action='store_true',
                        help='Traiter les images de la camera du haut.')
    action.add_argument('-l', '--lower', action='store_true',
                        help='Traiter les images de la camera du bas.')
    args = parser.parse_args()

    if args.simulation:
        env = 'Simulation'
    elif args.robot:
        env = 'Robot'
    elif args.genere:
        env = 'Genere'
    
    if args.upper:
        cfg.camera = "upper"
    else:
        cfg.camera = "lower"
    
    dossier_brut, dossier_PNG = get_dossiers(env)
    print("De " + dossier_brut + " vers " + dossier_PNG)
    brut_2_png(dossier_brut, dossier_PNG, True, env)

if __name__ == '__main__':
    main()
