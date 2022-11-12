import json

import os

def main():
    with open('stats_modeles_confidence_upper.json', 'r') as f:
        stats = json.load(f)

    pourcent_1 = "0.01"
    
    keys = [s for s in stats]
    keys = sorted(keys, key=lambda x:stats[x][pourcent_1]['vrai_positifs_acceptes'])
    print(f'Meilleur modele a {float(pourcent_1)*100}% de faux positifs : ' + keys[-1])
    for k in stats[keys[-1]][pourcent_1]:
        stats[keys[-1]][pourcent_1][k] = round(stats[keys[-1]][pourcent_1][k], 4)
    print(stats[keys[-1]][pourcent_1])

if __name__ == '__main__':
    main()
