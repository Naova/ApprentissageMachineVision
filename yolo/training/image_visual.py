import json
import matplotlib.pyplot as plt
import yolo.utils.args_parser as args_parser
from PIL import Image
import numpy as np
import random


def visualize_png(path_images):

    for i in path_images:
        x = Image.open(i.replace('YCbCr', 'RGB'))
        x = np.array(x)
        print(i)
        plt.imshow(x)
        plt.show()

def main():
    args = args_parser.parse_args_env_cam('Patate.')
    env = args_parser.set_config(args)
    with open('image_fn.json', 'r') as fn:
        fn_images_global = json.load(fn)
    with open('image_fp.json', 'r') as fp:
        fp_images_global = json.load(fp)

    for i in fp_images_global:
        random.shuffle(fp_images_global[i])
        visualize_png(fp_images_global[i])

if __name__ == '__main__':
    main()
