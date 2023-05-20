import numpy as np
import tqdm

import matplotlib.pyplot as plt

import yolo.utils.args_parser as args_parser
from yolo.training.clustering import read_rayons
from yolo.training.ball.train import load_model


def main():
    args = args_parser.parse_args_env_cam('')
    env = args_parser.set_config(args)
    modele = load_model(env=env)
    modele.summary()
    
    for j, layer in enumerate(modele.layers):
        if 'bias' in dir(layer):
            modele.layers[j].set_weights([np.ones(w.shape) for w in layer.weights[:-1]] + [np.zeros(layer.bias.shape)])

    y = 60
    x = 80
    
    resultat = np.zeros((120, 160))
    
    for y in tqdm.tqdm(range(120)):
        for x in tqdm.tqdm(range(160), leave=False):
            image = np.zeros((1, 120, 160, 3))
            image[0, y, x] = [1000, 1000, 1000]
            pred = modele.predict(image)
            if pred[0, 7, 10, 0] == 1:
                resultat[y, x] = 1
    plt.title('Champ receptif')
    plt.imshow(resultat)
    plt.show()
    
    rayons = read_rayons('Simulation')
    plt.title('Distribution des tailles de balles')
    plt.scatter(range(len(rayons)), [x*160 for x in sorted(rayons)])
    plt.show()
    
    image = np.zeros((120, 160))
    for i in range(15):
        for j in range(20):
            image[int(i * 120 / 15), :] = 1
            image[:, int(j * 160 / 20)] = 1
    plt.title('Grille de predictions')
    plt.imshow(image)
    plt.show()
    

if __name__ == '__main__':
    main()
