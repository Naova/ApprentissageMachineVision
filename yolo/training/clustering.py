from yolo.utils.configuration_provider import ConfigurationProvider as cfg_prov
import yolo.utils.args_parser as args_parser

from sklearn.cluster import KMeans

import numpy as np
import json
import random
from statistics import mean
import matplotlib.pyplot as plt

def read_rayons(env):
    image_width, image_height = cfg_prov.get_config().get_image_resolution(env)
    rayons = []
    with open(cfg_prov.get_config().get_labels_path(env), 'r') as fichier:
        labels = json.loads(fichier.read())
    for image_label in labels:
        if cfg_prov.get_config().detector == 'balles':
            objs = [label for label in labels[image_label] if label['categorie'] == 1]
        else:
            objs = [label for label in labels[image_label] if label['categorie'] == 2]
        for obj in objs:
            width = obj['right'] - obj['left']
            height = obj['bottom'] - obj['top']
            if cfg_prov.get_config().detector == 'balles':
                rayons.append(max(width / image_width, height / image_height) / 2)
            else:
                rayons.append(width)
    return rayons

def main():
    args = args_parser.parse_args_env_cam('Perform clustering on object size to select anchor boxes sizes.')
    env = args_parser.set_config(args)
    
    if cfg_prov.get_config().detector == 'balles':
        env = 'Simulation'
    else:
        env = 'Kaggle'
    
    rayons = read_rayons(env)

    if cfg_prov.get_config().detector == 'robots':
        #supprime les boites trop petites
        rayons = [r for r in rayons if r >= 25]
    
    km = KMeans(cfg_prov.get_config().get_nb_anchors())
    rayons = np.array(rayons).reshape(-1, 1)
    km.fit(rayons)
    anchors = km.cluster_centers_.tolist()
    rayons.reshape(len(rayons))
    rayons.sort()
    anchors = np.array(anchors)
    anchors = anchors.reshape(len(anchors))
    anchors = anchors.tolist()
    
    with open('rayons.csv', 'w') as f:
        for rayon in rayons:
            s = str(rayon)
            f.write(s.replace('.', ',') + '\n')
    print(anchors)


    if cfg_prov.get_config().detector == 'balles':
        plt.scatter(range(len(rayons)), rayons)
        plt.scatter(range(len(anchors)), anchors)
        plt.show()
    else:
        plt.scatter(range(len(rayons)), rayons)
        plt.scatter(range(len(anchors)), anchors)
        plt.show()
    with open(cfg_prov.get_config().get_anchors_path(), 'w') as anchors_file:
        anchors_file.write(json.dumps(anchors))

if __name__ == '__main__':
    main()
