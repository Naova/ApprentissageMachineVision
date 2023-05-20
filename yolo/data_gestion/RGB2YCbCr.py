import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os

from yolo.utils.configuration_provider import ConfigurationProvider as cfg_prov
import yolo.utils.args_parser as args_parser


def get_dossiers(env='Simulation'):
    return cfg_prov.get_config().get_dossier(env, 'RGB'), cfg_prov.get_config().get_dossier(env, 'YCbCr')

def rgb_2_ycbcr(path_entree, path_sortie, env):
    dossier_entree = Path(path_entree).glob('**/*/*/*')
    fichiers = [x.as_posix() for x in dossier_entree]
    if len(fichiers) == 0:
        dossier_entree = Path(path_entree).glob('**/*/*')
        fichiers = [x.as_posix() for x in dossier_entree]
    for fichier in tqdm(fichiers):
        dossier_sortie = fichier[:len(fichier)-fichier[::-1].index('/')].replace('RGB', 'YCbCr')
        if not os.path.exists(dossier_sortie):
            os.makedirs(dossier_sortie)
        if os.path.exists(fichier.replace('RGB', 'YCbCr')):
            continue
        image = Image.open(fichier)

        arr = np.array(image).astype('float')

        xform = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, 0.081]])
        arr = arr.dot(xform.T)
        arr[:,:,[1,2]] += 128
        np.putmask(arr, arr > 255, 255)
        np.putmask(arr, arr < 0, 0)

        Image.fromarray(arr.astype('uint8')).save(fichier.replace('RGB', 'YCbCr'))

def main():
    args = args_parser.parse_args_env_cam('Convertit toutes les images PNG RGB en YCbCr dans le dossier correspondant.',
                                         genere=True,
                                         hardnegative=True,
                                         testrobot=True)

    env = args_parser.set_config(args, False, True, False)
    
    dossier_RGB, dossier_YCBCr = get_dossiers(env)
    print("De " + dossier_RGB + " vers " + dossier_YCBCr)
    rgb_2_ycbcr(dossier_RGB, dossier_YCBCr, env)

if __name__ == '__main__':
    main()
