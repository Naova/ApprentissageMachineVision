import argparse

from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov

def parse_args_env_cam(description: str,
                        genere: bool = False,
                        hardnegative: bool = False,
                        testrobot: bool = False,
                        choosedetector: bool = True):
    parser = argparse.ArgumentParser(description=description)

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('-s', '--simulation', action='store_true',
                        help='Utiliser l\'environnement de la simulation.')
    action.add_argument('-r', '--robot', action='store_true',
                        help='Utiliser l\'environnement des robots.')
    if genere:
        action.add_argument('-g', '--genere', action='store_true',
                        help='Utiliser l\'environnement genere par CycleGAN.')
    if hardnegative:
        action.add_argument('-hn', '--hardnegative', action='store_true',
                        help='Utiliser les photos HardNegative.')
        action.add_argument('-nn', '--newnegative', action='store_true',
                        help='Utiliser les photos NewNegative.')
    if testrobot:
        action.add_argument('-tr', '--testrobot', action='store_true',
                        help='Utiliser les photos de tests des modeles de robot.')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('-u', '--upper', action='store_true',
                        help='Utiliser la camera du haut.')
    action.add_argument('-l', '--lower', action='store_true',
                        help='Utiliser la camera du bas.')
    if choosedetector:
        action = parser.add_mutually_exclusive_group(required=True)
        action.add_argument('-db', '--detect_balls', action='store_true',
                            help='Entrainer un detecteur de balles.')
        action.add_argument('-dr', '--detect_robots', action='store_true',
                            help='Entrainer un detecteur de robots.')

    return parser.parse_args()

def set_config(args, use_robot: bool = True, use_kaggle: bool = False, use_genere: bool = False):
    if args.detect_balls:
        cfg_prov.set_config('balles')
    else:
        cfg_prov.set_config('robots')
    if args.upper:
        cfg_prov.get_config().camera = "upper"
    else:
        cfg_prov.get_config().camera = "lower"
    if use_robot:
        if args.robot:
            return "Robot"
    else:
        if use_kaggle:
            return "Kaggle"
        if args.robot:
            return "Genere"
    if args.simulation:
        return "Simulation"
    if use_genere:
        if args.genere:
            return "Genere"
    if args.hardnegative:
        return "HardNegative"
    if args.newnegative:
        return "NewNegative"
    if args.testrobot:
        return "TestRobot"
    raise Exception('Pas d''environnement valide selectionne!')
