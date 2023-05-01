import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#import tensorflow.keras as keras
from tqdm import tqdm

from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov
from yolo.utils import args_parser

def get_images_test_YCbCr(env='TestRobot'):
    return cfg_prov.get_config().get_dossier(env, 'YCbCr')

# image.convert('L') avec RGB et non YCbCR :)

def YCbCr_level_test(path_entree, env):
    #ouvrir les images contenu dans les fichiers 
    dossier_entree = Path(path_entree).glob('**/*/*/*')
    fichiers = [x.as_posix() for x in dossier_entree]
    if len(fichiers) == 0:
        dossier_entree = Path(path_entree).glob('**/*/*')
        fichiers = [x.as_posix() for x in dossier_entree]

    # Recupere valeurs canal Y
    Y_level= []
    for fichier in tqdm(fichiers):
        image = Image.open(fichier)
        image_YCbCr = np.array(image)
        image_Y = image_YCbCr[:,:,0].astype(np.float32) / 255.0
        Y_level.append(image_Y.mean())
    
    Y_level = np.sort(Y_level)
    
    #AFFICHER AVEC MATPLOTLIB
    # Titres
    plt.title('Niveau de luminosité des images YCbCR')
    plt.xlabel('Images')
    plt.ylabel('Valeurs du canal Y')
    # Settings axes
    x = [i for i in range(len(Y_level))]
    plt.yticks(np.arange(0, 1.1, 0.1))
    # Affichage
    plt.grid(True)
    plt.plot(x,Y_level)
    plt.show()


def main():
    args = args_parser.parse_args_env_cam('Recuperer le niveau luminosité de toutes les images YCbCr dans le dossier correspondant.',
                                         #genere=True,
                                         #hardnegative=True,
                                         testrobot=True)

    env = args_parser.set_config(args)
    dossier_YCBCr = get_images_test_YCbCr(env)
    YCbCr_level_test(dossier_YCBCr,env)

if __name__ == '__main__':
    main()

