#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dec 20 17:39 2016

@author: Denis Tome'
"""

import __init__

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose
from lifting.utils import plot_pose_realtime

import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from os.path import dirname, realpath

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
#IMAGE_FILE_PATH = PROJECT_PATH + '/data/images/test_image.png'
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'

def main():
    global ax, fig, pose_estimator
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        exit()
    
    ret_val, image = cam.read()
    if not ret_val:
        print("Can't receive frame. Exiting ...")
        
    #image = cv2.imread(frame)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb

    # create pose estimator
    image_size = image.shape

    pose_estimator = PoseEstimator(image_size, SESSION_PATH, PROB_MODEL_PATH)

    # load model
    pose_estimator.initialise()

    cam.release()
    cv2.destroyAllWindows()

    # prepare for update
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    while True:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Cannot open camera after calibrating")
            exit()
        ret_val, image = cam.read()
        if not ret_val:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        try:
            # estimation
            #pose_2d, visibility, pose_3d = pose_estimator.estimate(image)

            # Show 2D and 3D poses
            #display_results(image, pose_2d, visibility, pose_3d)
            
            ani = animation.FuncAnimation(fig, func=animation_frame, interval=1000, repeat=False, fargs=(cam,))
            #for single_3D in pose_3d:
            #    # or plot_pose(Prob3dPose.centre_all(single_3D))
            #    plot_pose_realtime(single_3D)
            #del ani
            plt.show()

        except ValueError:
            print('No visible people in the image. Change CENTER_TR in packages/lifting/utils/config.py ...')

        if cv2.waitKey(1) == 27:
            break

    # close model
    pose_estimator.close()
    cam.release()
    cv2.destroyAllWindows()


def display_results(in_image, data_2d, joint_visibility, data_3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    #plt.figure()
    #draw_limbs(in_image, data_2d, joint_visibility)
    #plt.imshow(in_image)
    #plt.axis('off')

    # Show 3D poses
    for single_3D in data_3d:
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose(single_3D)

    plt.show()

def animation_frame(i, camtup):
    """animate plot"""
    #ax.cla()
    print(i)
    ret_val, image = camtup.read()
    pose_2d, visibility, pose_3d = pose_estimator.estimate(image)
    for single_3D in pose_3d:
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose_realtime(single_3D, ax, fig)
        print(single_3D)



if __name__ == '__main__':
    import sys
    sys.exit(main())