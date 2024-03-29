import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os

from yolo.utils.configuration_provider import ConfigurationProvider as cfg_prov
import yolo.utils.args_parser as args_parser


def get_dossiers(env='Simulation'):
    return cfg_prov.get_config().get_dossier(env, 'YCbCr'), cfg_prov.get_config().get_dossier(env, 'RGB')

def ycbcr_2_rgb(path_entree, path_sortie, env):
    dossier_entree = Path(path_entree).glob('**/*/*/*')
    fichiers = [x.as_posix() for x in dossier_entree]
    if len(fichiers) == 0:
        dossier_entree = Path(path_entree).glob('**/*/*')
        fichiers = [x.as_posix() for x in dossier_entree]
    for fichier in tqdm(fichiers):
        dossier_sortie = fichier[:len(fichier)-fichier[::-1].index('/')].replace('YCbCr', 'RGB')
        if not os.path.exists(dossier_sortie):
            os.makedirs(dossier_sortie)
        if os.path.exists(fichier.replace('YCbCr', 'RGB')):
            continue
        image = Image.open(fichier)

        arr = np.array(image).astype('float')

        xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
        arr[:,:,[1,2]] -= 128
        arr = arr.dot(xform.T)
        np.putmask(arr, arr > 255, 255)
        np.putmask(arr, arr < 0, 0)

        Image.fromarray(arr.astype('uint8')).save(fichier.replace('YCbCr', 'RGB'))

def main():
    args = args_parser.parse_args_env_cam('Convertit toutes les images PNG YCbCr en RGB dans le dossier correspondant.',
                                         genere=True,
                                         hardnegative=True,
                                         testrobot=True)

    env = args_parser.set_config(args)
    
    dossier_YCBCr, dossier_RGB = get_dossiers(env)
    print("De " + dossier_YCBCr + " vers " + dossier_RGB)
    ycbcr_2_rgb(dossier_YCBCr, dossier_RGB, env)

if __name__ == '__main__':
    main()
