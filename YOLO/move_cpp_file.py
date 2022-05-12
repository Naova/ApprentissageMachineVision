import sys
sys.path.insert(0,'..')

import config as cfg

import utils
import shutil

def main():
    args = utils.parse_args_env_cam('Deplace le fichier cpp a la bonne place dans NaovaCode.')
    env = utils.set_config(args)
    model_path = cfg.get_modele_path(env)
    source = 'cnn_'+model_path.replace('.h5', '.cpp')
    print(source)
    destination = f'../{cfg.naovaCodePath}/Src/Tools/NaovaTools/'
    print(destination)
    shutil.copy(source, destination)

if __name__ == '__main__':
    main()
