import train
import test_robot_model
import validate_hard_negative

import sys
sys.path.insert(0,'..')

import config as cfg

def main():
    while True:
        train.main()
        test_robot_model.main()
        validate_hard_negative.main()

if __name__ == '__main__':
    main()