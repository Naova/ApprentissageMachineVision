import numpy as np

from yolo.utils.configuration_provider import ConfigurationProvider as cfg_prov

def IoU(boite1, boite2):
    #gauche haut droite bas
    if boite1[2] < boite2[0] or boite1[0] > boite2[2] or boite1[1] > boite2[3] or boite1[3] < boite2[1]:
        return 0
    intersection = (min(boite1[2], boite2[2]) - max(boite1[0], boite2[0])) * (min(boite1[3], boite2[3]) - max(boite1[1], boite2[1]))
    union = ((boite1[2] - boite1[0]) * (boite1[3] - boite1[1])) + ((boite2[2] - boite2[0]) * (boite2[3] - boite2[1])) - intersection
    return intersection / union

def get_balle(y):
    resized_image_height, resized_image_width = cfg_prov.get_config().get_model_input_resolution()
    yolo_height, yolo_width = cfg_prov.get_config().get_model_output_resolution()
    ratio_x = resized_image_width / yolo_width
    ratio_y = resized_image_height / yolo_height
    coords = np.where(y[:,:,0] == y[:,:,0].max())
    for i, obj in enumerate(y[coords]):
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
        rayon = anchors[anchor_index] * resized_image_width
        left = int(center_x - rayon)
        top = int(center_y - rayon)
        right = int(center_x + rayon)
        bottom = int(center_y + rayon)
    return (left, top, right, bottom)

def iou_balles(pred, gt):
    boite1 = get_balle(pred)
    boite2 = get_balle(gt)
    return IoU(boite1, boite2)

