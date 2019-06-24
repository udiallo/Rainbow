import monodepth.monodepth_inference as depth_estimation
from obstacle_tower_od import ObjectDetection
import torch
import numpy as np

def prepare_input(rgb, depth_model, OD):

    # Object detection and monodepth


    depthmap = depth_model.mono_inference(rgb, debug=False)
    all_objects = OD.detect_objects(rgb, im_save=True, dir='', im_id=0)
    objects = []
    for obj in all_objects:
        if (obj[1] == "exit_door"):
            center_x, center_y = calculate_center(obj[0][0], obj[0][1], obj[0][2], obj[0][3])
            z_mean = calculate_z(obj[0][0], obj[0][1], obj[0][2], obj[0][3], depthmap)
            x_y_z = np.array([center_x, center_y, z_mean, 1])
            objects.append(x_y_z)
        elif (obj[1] == "door"):
            center_x, center_y = calculate_center(obj[0][0], obj[0][1], obj[0][2], obj[0][3])
            z_mean = calculate_z(obj[0][0], obj[0][1], obj[0][2], obj[0][3])
            x_y_z = np.array([center_x, center_y, z_mean, 1])
            objects.append(x_y_z)
        else:
            x_y_z = np.array([0, 0, 0, 0])
            objects.append(x_y_z)
    # loop through objects and for every exit_door, take x, y, z, 1 values
    # if no object detected then 0,0,0 as x,y,z,0
    # create mask with depth image,everything which is not in the bounding box gets z values 0,
    # 2. option: take mean value of submatrix
    # calculate index of bounding box: leftcorner x value * 128, maybe same for lowerright corner * 128

    #loop through objects, check which array I need, create submatrix and claculate z mean value
    # give into the model x,y,z,1 or 0,0,0,0


    return objects, depthmap


def calculate_center(upperb, leftb, lowerb, rightb):
    center_x = 0.5*rightb + leftb
    center_y = 0.5*lowerb + upperb

    return center_x, center_y

def calculate_z(upperb, leftb, lowerb, rightb, depthmap):



    pixel_left_x = leftb * 128
    pixel_up_y = upperb * 128
    pixel_right_x = rightb * 128
    pixel_low_y = lowerb * 128

    submatrix = depthmap[int(round(pixel_left_x)):int(round(pixel_right_x)), int(round(pixel_up_y)):int(round(pixel_low_y))]
    z_mean = submatrix.mean()

    return z_mean