import sys
sys.path.insert(0,'..')
import config as cfg

import json
import random
from statistics import mean

def read_rayons(path_etiquettes):
    rayons = []
    with open(path_etiquettes, 'r') as fichier:
        labels = json.loads(fichier.read())
        for image_label in labels:
            balles = [label for label in labels[image_label] if label['categorie'] == 1]
            for balle in balles:
                width = balle['right'] - balle['left']
                height = balle['bottom'] - balle['top']
                rayons.append(max(width / cfg.image_width, height / cfg.image_height) / 2)
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

def kmean(population, k = 10):
    clusters = random.choices(population, k=k)

    while True:
        previous_clusters = clusters.copy()
        values = assign_values(clusters, population)
        clusters = [recalculate_mean(values[c]) for c in clusters]

        if previous_clusters == clusters:
            break
    return sorted(clusters)

if __name__ == '__main__':
    rayons = read_rayons(cfg.json_etiquettes)
    anchors = kmean(rayons, cfg.yolo_nb_anchors)
    print(anchors)
    with open(cfg.yolo_anchors_path, 'w') as anchors_file:
        anchors_file.write(json.dumps(anchors))