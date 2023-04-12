import json
import matplotlib.pyplot as plt
import yolo.utils.args_parser as args_parser
from PIL import Image
import numpy as np
from collections import Counter


def visualize_png(path_image):
    x = Image.open(path_image.replace('YCbCr', 'RGB'))
    x = np.array(x)
    print(path_image)
    plt.imshow(x)
    plt.show()
    input('Appuyez sur Entrer pour continuer...')

def count_images(data):
    file_counter = Counter()
    for model_name in data:
        file_counter.update(data[model_name])
    return file_counter

def main():
    args = args_parser.parse_args_env_cam('Permet de visualiser les faux positifs et faux negatifs courants rencontres durant les tests de modeles.')
    env = args_parser.set_config(args)
    with open('image_fn.json', 'r') as fn:
        fn_images = json.load(fn)
    with open('image_fp.json', 'r') as fp:
        fp_images = json.load(fp)
    
    fn_images = count_images(fn_images)
    fp_images = count_images(fp_images)

    plt.ion()

    print("Faux positifs", str(len(fp_images)))
    for image_path, i in fp_images.most_common():
        if i > 1:
            print(i)
            visualize_png(image_path)
    print("Faux Negatifs", str(len(fn_images)))
    for image_path, i in fn_images.most_common():
        if i > 1:
            print(i)
            visualize_png(image_path)

if __name__ == '__main__':
    main()
