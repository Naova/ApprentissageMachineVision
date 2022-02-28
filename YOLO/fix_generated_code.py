
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

def change_variable_access_method(full_text: str, variable: str):
    var_index = full_text.index(variable+'[')
    if variable == 'x_0':
        shape = (*cfg.get_resized_image_resolution(), 3)
    else:
        second_occurence = full_text.split(variable)[2]
        indice_name = second_occurence.split('][')[1]
        value = int(full_text.split(indice_name+' <')[1].split(';')[0])
        shape = (-1, value, cfg.get_nb_anchors() + 3)
    end_of_text = full_text[var_index:]
    next_bracket = end_of_text.index(']')
    full_text = full_text[:var_index] + end_of_text.replace('][', f'*{shape[1]}*{shape[2]} + ', 1)
    end_of_text = full_text[var_index:]
    full_text = full_text[:var_index] + end_of_text.replace('][', f'*{shape[2]} + ', 1)
    return full_text

def change_arguments_to_simple_pointers(full_text: str):
    function_beginning = full_text.index('void cnn')
    end_of_text = full_text[function_beginning:]
    for _ in range(3):
        end_of_text = remove_next_bracket_pair(end_of_text)
    next_float = end_of_text.index('float ')
    variables_declaration = end_of_text[:next_float] + 'const '
    x_0_index = end_of_text.index('x_0')
    variables_declaration += end_of_text[next_float:x_0_index] + '*'
    variables_declaration += end_of_text[x_0_index:]
    full_text = full_text[:function_beginning] + variables_declaration
    #a partir d'ici, la declaration de la fonction est OK
    output_variable = end_of_text[x_0_index + 12:end_of_text.index(')')]
    full_text = change_variable_access_method(full_text, 'x_0')
    full_text = change_variable_access_method(full_text, output_variable)
    return full_text

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