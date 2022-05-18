from skimage.draw import rectangle_perimeter

import matplotlib.pyplot as plt

import numpy as np

import argparse

import sys
sys.path.insert(0,'..')
import config as cfg


def parse_args_env_cam(description: str, genere: bool = False, hardnegative: bool = False):
    parser = argparse.ArgumentParser(description=description)

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('-s', '--simulation', action='store_true',
                        help='Utiliser l\'environnement de la simulation.')
    action.add_argument('-r', '--robot', action='store_true',
                        help='Utiliser l\'environnement des robots.')
    if genere:
        action.add_argument('-g', '--genere', action='store_true',
                        help='Utiliser l\'environnement genere par CycleGAN.')
    if hardnegative:
        action.add_argument('-hn', '--hardnegative', action='store_true',
                        help='Utiliser les photos HardNegative.')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('-u', '--upper', action='store_true',
                        help='Utiliser la camera du haut.')
    action.add_argument('-l', '--lower', action='store_true',
                        help='Utiliser la camera du bas.')

    return parser.parse_args()

def set_config(args, use_robot: bool = True, use_genere: bool = False):
    if args.upper:
        cfg.camera = "upper"
    else:
        cfg.camera = "lower"
    if use_robot:
        if args.robot:
            return "Robot"
    else:
        if args.robot:
            return "Genere"
    if args.simulation:
        return "Simulation"
    elif use_genere:
        if args.genere:
            return "Genere"
    if args.hardnegative:
        return "HardNegative"
    raise Exception('Pas d''environnement valide selectionne!')

def draw_rectangle_on_image(input_image, yolo_output, coords):
    resized_image_height, resized_image_width = cfg.get_resized_image_resolution()
    yolo_height, yolo_width = cfg.get_yolo_resolution()
    ratio_x = resized_image_width / yolo_width
    ratio_y = resized_image_height / yolo_height
    for i, obj in enumerate(yolo_output[coords]):
        center_x = (coords[1][i] + obj[1]) * ratio_x
        center_y = (coords[0][i] + obj[2]) * ratio_y
        anchor_index = np.where(obj[3:]==obj[3:].max())[0][0]
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

def non_max_suppression(prediction: np.array):
    #treshold
    coords = treshold_coord(prediction[:,:,0], 0.2)

    resized_image_height, resized_image_width = cfg.get_resized_image_resolution()
    yolo_height, yolo_width = cfg.get_yolo_resolution()
    ratio_x = resized_image_width / yolo_width
    ratio_y = resized_image_height / yolo_height

    boxes = np.empty((0, 5))
    #define boxes
    for i, j in zip(coords[0], coords[1]):
        confidence = prediction[i,j,0]
        anchor = cfg.get_anchors()[np.where(prediction==max(prediction[i,j,3:]))[2][0]-3]
        rayon = anchor * resized_image_width
        center_x = (i + prediction[i,j,1]) * ratio_x
        center_y = (j + prediction[i,j,2]) * ratio_y
        left = int(center_x - rayon)
        right = int(center_x + rayon)
        top = int(center_y - rayon)
        bottom = int(center_y + rayon)
        box = [confidence, left, top, right, bottom]
        boxes = np.concatenate((boxes, np.array([box])), axis=0)

    #merge NMS
    nb_boxes = len(boxes)
    for i in range(nb_boxes):
        if i >= len(boxes):
            break
        box1 = boxes[i]
        for j in range(nb_boxes):
            if i == j:
                continue
            if j >= len(boxes):
                break
            box2 = boxes[j]
            #check overlap
            
            #merge

            #remove boxes
    breakpoint()
    return coords
    
