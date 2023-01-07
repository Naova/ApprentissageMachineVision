import shutil

from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov
import yolo.config as cfg_global
import yolo.utils.args_parser as args_parser


def main():
    args = args_parser.parse_args_env_cam('Deplace le fichier cpp a la bonne place dans NaovaCode.')
    env = args_parser.set_config(args)
    model_path = cfg_prov.get_config().get_modele_path(env)
    source = 'cnn_'+model_path.replace('.h5', '.cpp')
    print(source)
    destination = f'{cfg_global.naovaCodePath}/Src/Tools/NaovaTools/'
    destination += source.replace('modele_balles', 'yolo_modele')
    print(destination)
    shutil.copy(source, destination)

if __name__ == '__main__':
    main()
