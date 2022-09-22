import shutil

import yolo.config as cfg
import yolo.utils.args_parser as args_parser


def main():
    args = args_parser.parse_args_env_cam('Deplace le fichier cpp a la bonne place dans NaovaCode.')
    env = args_parser.set_config(args)
    model_path = cfg.get_modele_path(env)
    source = 'cnn_'+model_path.replace('.h5', '.cpp')
    print(source)
    destination = f'{cfg.naovaCodePath}/Src/Tools/NaovaTools/'
    print(destination)
    shutil.copy(source, destination)

if __name__ == '__main__':
    main()
