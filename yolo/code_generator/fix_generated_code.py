from yolo.utils.configuration_provider import ConfigurationProvider as cfg_prov

import yolo.utils.args_parser as args_parser


def remove_unnecessary_code(full_text: str):
    """
    Le code genere est plein de calculs inutiles.
    Disons que c'est surtout une question d'esthetique et de lisibilite.
    """
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
        shape = (*cfg_prov.get_config().get_model_input_resolution(), 3)
    else:
        second_occurence = full_text.split(variable)[2]
        indice_name = second_occurence.split('][')[1]
        value = int(full_text.split(indice_name+' <')[1].split(';')[0])
        shape = (-1, value, 5 + cfg_prov.get_config().get_nb_anchors())
    end_of_text = full_text[var_index:]
    full_text = full_text[:var_index] + end_of_text.replace('][', f'*{shape[1]}*{shape[2]} + ', 1)
    end_of_text = full_text[var_index:]
    full_text = full_text[:var_index] + end_of_text.replace('][', f'*{shape[2]} + ', 1)
    return full_text

def add_warning_removal(full_text: str):
    full_text = """#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#pragma clang diagnostic ignored "-Wconversion"
""" + full_text + "\n#pragma clang diagnostic pop\n"

    return full_text

def change_arguments_to_simple_pointers(full_text: str):
    """
    Les parametres de la fonction principale sont declares comme des tableaux a trois dimensions (par exemple: float x_0[160][120][3])
    On souhaite plutot les avoir comme un simple pointeur  -> float * x_0
    On doit alors y acceder en faisant les mathematiques nous-memes.
    Par exemple:
        x_0[i][j][k] //avec j < 120 et k < 3
    devient :
        x_0[i * 120 * 3 + j * 3 + k]
    """
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
    
    if '#pragma clang' in full_text:
        return False
    
    full_text = remove_unnecessary_code(full_text)
    full_text = change_arguments_to_simple_pointers(full_text)
    full_text = add_warning_removal(full_text)
    
    with open(file_path, 'w') as f:
        f.write(full_text)
    return True

def main():
    args = args_parser.parse_args_env_cam('Modifie de petites choses dans le code C++ genere par NNCG pour pouvoir facilement l\'exporter dans NaovaCode.')
    env = args_parser.set_config(args)
    model_path = cfg_prov.get_config().get_modele_path(env).replace('.h5', '.cpp')
    model_path = f'cnn_{model_path}'
    if fix_file(file_path=model_path):
        print(f'{model_path} file fixed!')
    else:
        print(f'{model_path} is already fixed!')

if __name__ == '__main__':
    main()
