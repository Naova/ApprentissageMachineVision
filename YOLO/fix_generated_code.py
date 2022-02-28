
import argparse
import sys

from numpy import full, full_like
sys.path.insert(0,'..')

import config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Train the network given ')

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('-s', '--simulation', action='store_true',
                        help='Entrainer pour l\'environnement de la simulation.')
    action.add_argument('-r', '--robot', action='store_true',
                        help='Entrainer pour l\'environnement des robots.')
    action.add_argument('-d', '--dummy', action='store_true',
                        help='Dummy model.')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('-u', '--upper', action='store_true',
                        help='Entrainer pour la camera du haut.')
    action.add_argument('-l', '--lower', action='store_true',
                        help='Entrainer pour la camera du bas.')

    return parser.parse_args()

def set_config(args):
    if args.upper:
        cfg.camera = "upper"
    else:
        cfg.camera = "lower"
    if args.simulation:
        env = "Simulation"
    else:
        env = "Genere"
    return env

def remove_unnecessary_code(full_text: str):
    full_text = full_text.replace(' + 0]', ']')
    full_text = full_text.replace(' - 0;', ';')
    full_text = full_text.replace(' / 1]', ']')
    return full_text

def remove_next_bracket_pair(text: str):
    next_open_bracket = text.index('[')
    next_closing_bracket = text.index(']')
    return text[:next_open_bracket] + text[next_closing_bracket + 1:]

def change_arguments_to_simple_pointers(full_text: str):
    function_beginning = full_text.index('void cnn')
    #next_bracket = full_text[function_beginning:].index('[') + function_beginning
    end_of_text = full_text[function_beginning:]
    for _ in range(6):
        end_of_text = remove_next_bracket_pair(end_of_text)
    return full_text[:function_beginning] + end_of_text

def fix_file(file_path:str):
    full_text = ''
    with open(file_path, 'r') as f:
        full_text = f.read()
    
    full_text = remove_unnecessary_code(full_text)
    full_text = change_arguments_to_simple_pointers(full_text)
    
    with open(file_path, 'w') as f:
        f.write(full_text)

def main():
    args = parse_args()
    env = set_config(args)
    model_path = cfg.get_modele_path(env).replace('.h5', '.cpp')
    model_path = f'cnn_{model_path}'
    fix_file(file_path=model_path)

if __name__ == '__main__':
    main()