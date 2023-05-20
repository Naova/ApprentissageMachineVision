from keras_flops import get_flops

from yolo.utils import args_parser
from yolo.training.ball.train import load_model

def main():
    args = args_parser.parse_args_env_cam('Test the yolo model on a bunch of test images and output stats.')
    env = args_parser.set_config(args)
    
    modele = load_model(env=env)
    modele.summary()
    
    flops = get_flops(modele, batch_size=1)
    
    print(f"FLOPS: {flops / 10 ** 9:.03} G")

if __name__ == '__main__':
    main()
