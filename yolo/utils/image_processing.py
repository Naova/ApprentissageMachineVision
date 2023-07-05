from skimage.draw import rectangle_perimeter
import matplotlib.pyplot as plt
import numpy as np

from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov

def draw_rectangle_on_image(input_image, yolo_output, coords):
    resized_image_height, resized_image_width = cfg_prov.get_config().get_model_input_resolution()
    yolo_height, yolo_width = cfg_prov.get_config().get_model_output_resolution()
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
        anchors = cfg_prov.get_config().get_anchors()
        if cfg_prov.get_config().detector == 'balles':
            rayon = anchors[anchor_index] * resized_image_width
            left = int(center_x - rayon)
            top = int(center_y - rayon)
            right = int(center_x + rayon)
            bottom = int(center_y + rayon)
        else:
            input_resolution = cfg_prov.get_config().get_image_resolution('Kaggle')
            box = (
                anchors[anchor_index][0] * resized_image_width / input_resolution[0],
                anchors[anchor_index][1] * resized_image_width / input_resolution[1]
            )
            left = int(center_x - box[0]/2)
            right = int(center_x + box[0]/2)
            if cfg_prov.get_config().detector == 'robots':
                top = int(center_y - box[1])
                bottom = int(center_y)
            else:
                top = int(center_y - box[1]/2)
                bottom = int(center_y + box[1]/2)
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


def display_model_prediction(prediction, wanted_prediction, prediction_on_image, wanted_output, filename):
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
    plt.savefig('predictions/' + filename, dpi=300)
    plt.clf()

def generate_prediction_image(prediction, x_test, y_test, filename):
    coords = treshold_coord(prediction[:,:,0], 0.45)
    prediction_on_image = draw_rectangle_on_image(x_test.copy(), prediction, coords)
    coords = treshold_coord(y_test[:,:,0])
    wanted_output = draw_rectangle_on_image(x_test, y_test, coords)
    display_model_prediction(prediction[:,:,0], y_test[:,:,0], prediction_on_image, wanted_output, filename)

