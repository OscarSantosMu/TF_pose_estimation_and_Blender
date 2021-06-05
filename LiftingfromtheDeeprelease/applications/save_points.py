#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dec 20 17:39 2016

@author: Oscar Alberto Santos Muñoz & Uriel Haile Hernández Belmonte'
"""

import __init__

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose

import cv2
import numpy as np
import json, codecs
import matplotlib.pyplot as plt
from os.path import dirname, realpath, join

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
IMAGE_FILE_PATH = PROJECT_PATH + '/data/images/p1.jpg'
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'


def main():
    image = cv2.imread(IMAGE_FILE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb

    # create pose estimator
    image_size = image.shape

    pose_estimator = PoseEstimator(image_size, SESSION_PATH, PROB_MODEL_PATH)

    # load model
    pose_estimator.initialise()

    try:
        # estimation
        pose_2d, visibility, pose_3d = pose_estimator.estimate(image)

        # Show 2D and 3D poses
        #np.set_printoptions(formatter={'float_kind':'{:f}'.format})
        #display_results(image, pose_2d, visibility, pose_3d)
        normalized_pose_3d = 2.*(pose_3d - np.min(pose_3d))/np.ptp(pose_3d)-1
        display_results(image, pose_2d, visibility, normalized_pose_3d)
        
        lst = pose_3d.tolist()
        print(lst)
        n_lst = normalized_pose_3d.tolist()
        pose_3d_xyz = np.array(pose_3d[0]).transpose()
        print(pose_3d_xyz)
        print(type(pose_3d_xyz))
        print(pose_3d_xyz.shape)
        normalized_pose_3d_xyz = 1.7728*100*(pose_3d_xyz - np.min(pose_3d_xyz))/np.ptp(pose_3d_xyz)
        print(np.max(pose_3d_xyz))
        print(np.min(pose_3d_xyz))
        print(np.ptp(pose_3d_xyz))
        print(normalized_pose_3d_xyz)
        n_lst_xyz = normalized_pose_3d_xyz.tolist()
        #with open("points.json", 'w') as f:
                # indent=2 is not needed but makes the file human-readable
        json.dump(n_lst, codecs.open(join(DIR_PATH,"points.json"), 'w', encoding='utf-8'), separators=(',', ':'), indent=2) 
        json.dump(n_lst_xyz, codecs.open(join(DIR_PATH,"points_xyz.json"), 'w', encoding='utf-8'), separators=(',', ':'), indent=2)

    except ValueError:
        print('No visible people in the image. Change CENTER_TR in packages/lifting/utils/config.py ...')

    # close model
    pose_estimator.close()


def display_results(in_image, data_2d, joint_visibility, data_3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    plt.figure()
    draw_limbs(in_image, data_2d, joint_visibility)
    plt.imshow(in_image)
    plt.axis('off')

    # Show 3D poses
    for single_3D in data_3d:
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        print(single_3D)
        plot_pose(single_3D)

    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(main())
