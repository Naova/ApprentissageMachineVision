import json

import os

def main():
    with open('stats_modeles_confidence_lower.json', 'r') as f:
        stats = json.load(f)

    pourcent_1 = "0.01"
    
    keys = [s for s in stats]
    keys = sorted(keys, key=lambda x:stats[x][pourcent_1]['recall'])
    print(f'Meilleur modele a {float(pourcent_1)*100}% de faux positifs : ' + keys[-1])
    for k in stats[keys[-1]][pourcent_1]:
        stats[keys[-1]][pourcent_1][k] = round(stats[keys[-1]][pourcent_1][k], 4)
    for key in keys[-15:]:
        print(key, stats[key][pourcent_1])

if __name__ == '__main__':
    main()
