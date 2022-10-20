import yolo.training.ball.train as train
import yolo.training.test_all_models as test_all_models
import yolo.training.ball.validate_hard_negative as validate_hard_negative

def main():
    while True:
        train.main()
        test_all_models.main()
        validate_hard_negative.main()

if __name__ == '__main__':
    main()
