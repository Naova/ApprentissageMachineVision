import json
import matplotlib.pyplot as plt
import yolo.utils.args_parser as args_parser
from PIL import Image
import numpy as np


def visualize_png(path_images):
    for i in path_images:
        x = Image.open(i.replace('YCbCr', 'RGB'))
        x = np.array(x)
        print(i)
        plt.imshow(x)
        plt.show()
        input('Appuyez sur Entrer pour continuer...')

def main():
    args = args_parser.parse_args_env_cam('Patate.')
    env = args_parser.set_config(args)
    with open('image_fn.json', 'r') as fn:
        fn_images_global = json.load(fn)
    with open('image_fp.json', 'r') as fp:
        fp_images_global = json.load(fp)
    
    plt.ion()

    print("Faux positifs", str(len(fp_images_global)))
    for i in fp_images_global:
        visualize_png(fp_images_global[i])
    print("Faux Negatifs", str(len(fn_images_global)))
    for i in fn_images_global:
        visualize_png(fn_images_global[i])

if __name__ == '__main__':
    main()
