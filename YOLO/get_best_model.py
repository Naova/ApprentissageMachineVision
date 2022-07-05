import json

import os

def main():
    with open('stats_modeles_confidence_upper.json', 'r') as f:
        stats = json.load(f)
    
    pourcent_1 = 0.02
    pourcent_2 = 0.015
    pourcent_3 = 0.01
    
    keys = [s for s in stats]
    keys = sorted(keys, key=lambda x:stats[x][pourcent_1]['vrai_positifs_acceptes'])
    print('Meilleur modele à 0.5% de faux positifs : ' + keys[-1])
    for k in stats[keys[-1]][pourcent_1]:
        stats[keys[-1]][pourcent_1][k] = round(stats[keys[-1]][pourcent_1][k], 4)
    print(stats[keys[-1]][pourcent_1])
    keys = sorted(keys, key=lambda x:stats[x][pourcent_2]['vrai_positifs_acceptes'])
    for k in stats[keys[-1]][pourcent_2]:
        stats[keys[-1]][pourcent_2][k] = round(stats[keys[-1]][pourcent_2][k], 4)
    print('Meilleur modele à 0.2% de faux positifs : ' + keys[-1])
    print(stats[keys[-1]][pourcent_2])
    keys = sorted(keys, key=lambda x:stats[x][pourcent_3]['vrai_positifs_acceptes'])
    for k in stats[keys[-1]][pourcent_3]:
        stats[keys[-1]][pourcent_3][k] = round(stats[keys[-1]][pourcent_3][k], 4)
    print('Meilleur modele à 0.1% de faux positifs : ' + keys[-1])
    print(stats[keys[-1]][pourcent_3])

if __name__ == '__main__':
    main()
