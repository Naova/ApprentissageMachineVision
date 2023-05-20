from yolo.utils.configuration_provider import ConfigurationProvider as cfg_prov
import yolo.utils.args_parser as args_parser

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
                rayons.append((width, height))
    return rayons

def distance_balle(x1, x2):
    return abs(x1 - x2)

def distance_robot(x1, x2):
    return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])

def assign_values(clusters:list, population:list):
    values = {} # {cluster1:[pop1, pop2, pop3, ...], cluster2:[...], ...}
    for c in clusters:
        values[c] = []
    
    for p in population:
        if cfg_prov.get_config().detector == 'balles':
            distances = [distance_balle(p, c) for c in clusters]
        else:
            distances = [distance_robot(p, c) for c in clusters]
        closer_cluster = clusters[distances.index(min(distances))]
        values[closer_cluster].append(p)

    return values

def recalculate_mean_balle(population:list):
    return mean(population)

def recalculate_mean_robot(population:list):
    return (mean([x[0] for x in population]), mean([x[1] for x in population]))

def kmean(population, k):
    clusters = random.choices(population, k=k)

    while True:
        previous_clusters = clusters.copy()
        values = assign_values(clusters, population)
        if cfg_prov.get_config().detector == 'balles':
            clusters = [recalculate_mean_balle(values[c]) for c in clusters]
        else:
            clusters = [recalculate_mean_robot(values[c]) for c in clusters]

        if previous_clusters == clusters:
            break
    return sorted(clusters)

def main():
    args = args_parser.parse_args_env_cam('Perform clustering on object size to select anchor boxes sizes.')
    env = args_parser.set_config(args)
    
    if cfg_prov.get_config().detector == 'balles':
        env = 'Simulation'
    else:
        env = 'Kaggle'
    
    rayons = read_rayons(env)

    anchors = kmean(rayons, cfg_prov.get_config().get_nb_anchors())
    rayons.sort()
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
        plt.scatter([x[0] for x in rayons], [x[1] for x in rayons])
        plt.scatter([a[0] for a in anchors], [a[1] for a in anchors])
        plt.show()
    with open(cfg_prov.get_config().get_anchors_path(), 'w') as anchors_file:
        anchors_file.write(json.dumps(anchors))

if __name__ == '__main__':
    main()
