from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov
import yolo.utils.args_parser as args_parser

import json
import random
from statistics import mean

def read_rayons(path_etiquettes):
    image_width, image_height = cfg_prov.get_config().get_image_resolution()
    rayons = []
    with open(path_etiquettes, 'r') as fichier:
        labels = json.loads(fichier.read())
        for image_label in labels:
            balles = [label for label in labels[image_label] if label['categorie'] == 1]
            for balle in balles:
                width = balle['right'] - balle['left']
                height = balle['bottom'] - balle['top']
                rayons.append(max(width / image_width, height / image_height) / 2)
    return rayons

def distance(x1, x2):
    return abs(x1 - x2)

def assign_values(clusters:list, population:list):
    values = {} # {cluster1:[pop1, pop2, pop3, ...], cluster2:[...], ...}
    for c in clusters:
        values[c] = []
    
    for p in population:
        distances = [distance(p, c) for c in clusters]
        closer_cluster = clusters[distances.index(min(distances))]
        values[closer_cluster].append(p)

    return values

def recalculate_mean(population:list):
    return mean(population)

def kmean(population, k):
    clusters = random.choices(population, k=k)

    while True:
        previous_clusters = clusters.copy()
        values = assign_values(clusters, population)
        clusters = [recalculate_mean(values[c]) for c in clusters]

        if previous_clusters == clusters:
            break
    return sorted(clusters)

def main(camera):
    args = args_parser.parse_args_env_cam('Perform clustering on object size to select anchor boxes sizes.')
    env = args_parser.set_config(args)
    cfg_prov.get_config().camera = camera
    
    rayons = read_rayons(cfg_prov.get_config().get_labels_path('Simulation'))
    anchors = kmean(rayons, cfg_prov.get_config().get_nb_anchors())
    rayons.sort()
    with open('rayons.csv', 'w') as f:
        for rayon in rayons:
            s = str(rayon)
            f.write(s.replace('.', ',') + '\n')
    print(anchors)
    with open(cfg_prov.get_config().get_anchors_path(), 'w') as anchors_file:
        anchors_file.write(json.dumps(anchors))

if __name__ == '__main__':
    main('upper')
    main('lower')
