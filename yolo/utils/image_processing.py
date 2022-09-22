from skimage.draw import rectangle_perimeter
import matplotlib.pyplot as plt
import numpy as np

import yolo.training.ball.config as cfg

def draw_rectangle_on_image(input_image, yolo_output, coords):
    resized_image_height, resized_image_width = cfg.get_resized_image_resolution()
    yolo_height, yolo_width = cfg.get_yolo_resolution()
    ratio_x = resized_image_width / yolo_width
    ratio_y = resized_image_height / yolo_height
    for i, obj in enumerate(yolo_output[coords]):
        if obj[1] > obj[2]:
            x = 0.25
        else:
            x = 0.75
        if obj[3] > obj[4]:
            y = 0.25
        else:
            y = 0.75
        center_x = (coords[1][i] + x) * ratio_x
        center_y = (coords[0][i] + y) * ratio_y
        anchor_index = np.where(obj[5:]==obj[5:].max())[0][0]
        anchors = cfg.get_anchors()
        rayon = anchors[anchor_index] * resized_image_width
        left = int(center_x - rayon)
        top = int(center_y - rayon)
        right = int(center_x + rayon)
        bottom = int(center_y + rayon)
        rect = rectangle_perimeter((top, left), (bottom, right), shape=(resized_image_height, resized_image_width), clip=True)
        input_image[rect] = 1
    return input_image

def ycbcr2rgb(img_ycbcr:np.array):
    #convertion en RGB
    img = img_ycbcr*255
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    img[:,:,[1,2]] -= 128
    img = img.dot(xform.T)
    np.putmask(img, img > 255, 255)
    np.putmask(img, img < 0, 0)
    return img/255

def treshold_coord(arr, treshold = 0.5):
    return np.where(arr >= treshold)

def n_max_coord(arr, n = 3):
    arr_2 = arr.flatten().copy()
    arr_2.sort()
    coords = np.empty((2, 0))
    for value in arr_2[-n:]:
        coords = np.concatenate((coords, np.where(arr == value)), axis=1)
    return (coords.astype(np.uint8)[0], coords.astype(np.uint8)[1])

def display_yolo_rectangles(input_image, yolo_output):
    coord = np.where(yolo_output[:,:,0] > 0.3)
    input_image = draw_rectangle_on_image(input_image, yolo_output, coord)
    plt.imshow(input_image)
    plt.show()


def display_model_prediction(prediction, wanted_prediction, prediction_on_image, wanted_output, save_to_file_name = None):
    fig = plt.figure()
    fig.add_subplot(2, 2, 1)
    plt.imshow(prediction)
    plt.title('model output')
    plt.colorbar(orientation='horizontal')
    fig.add_subplot(2, 2, 2)
    plt.imshow(wanted_prediction)
    plt.title('ground truth')
    plt.colorbar(orientation='horizontal')
    fig.add_subplot(2, 2, 3)
    plt.imshow(prediction_on_image)
    plt.title('model output on image')
    fig.add_subplot(2, 2, 4)
    plt.imshow(wanted_output)
    plt.title('ground truth on image')
    if save_to_file_name:
        plt.savefig('predictions/' + save_to_file_name, dpi=300)
    plt.show()

def generate_prediction_image(prediction, x_test, y_test, prediction_number = None):
    coords = n_max_coord(prediction[:,:,0], 1)
    prediction_on_image = draw_rectangle_on_image(ycbcr2rgb(x_test.copy()), prediction, coords)
    coords = treshold_coord(y_test[:,:,0])
    wanted_output = draw_rectangle_on_image(ycbcr2rgb(x_test.copy()), y_test, coords)
    display_model_prediction(prediction[:,:,0], y_test[:,:,0], prediction_on_image, wanted_output, 'prediction_' + str(prediction_number) + '.png')

