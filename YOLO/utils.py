from skimage.draw import rectangle_perimeter

import matplotlib.pyplot as plt

import numpy as np

import sys
sys.path.insert(0,'..')
import config as cfg

def draw_rectangle_on_image(input_image, yolo_output, coords):
    ratio_x = cfg.image_width / cfg.yolo_width
    ratio_y = cfg.image_height / cfg.yolo_height
    for i, obj in enumerate(yolo_output[coords]):
        center_x = (coords[1][i] + obj[1]) * ratio_x
        center_y = (coords[0][i] + obj[2]) * ratio_y
        anchor_index = np.where(obj[3:]==obj[3:].max())[0][0]
        anchors = cfg.get_anchors()
        rayon = anchors[anchor_index] * cfg.image_width
        left = int(center_x - rayon)
        top = int(center_y - rayon)
        right = int(center_x + rayon)
        bottom = int(center_y + rayon)
        rect = rectangle_perimeter((top, left), (bottom, right), shape=(cfg.image_height, cfg.image_width), clip=True)
        input_image[rect] = 1
    return input_image

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