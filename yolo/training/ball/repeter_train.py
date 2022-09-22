import yolo.training.ball.train as train
import yolo.training.ball.test_robot_model as test_robot_model
import yolo.training.ball.validate_hard_negative as validate_hard_negative



import yolo.training.ball.config as cfg

def main():
    while True:
        train.main()
        test_robot_model.main()
        validate_hard_negative.main()

if __name__ == '__main__':
    main()
