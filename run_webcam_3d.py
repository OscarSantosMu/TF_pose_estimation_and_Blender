import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.utils.prob_model import Prob3dPose
from lifting.utils import plot_pose

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def get_keypoints():
    image_h, image_w = image.reshape[:2]
    standard_w = 640
    standard_h = 480

    pose_2d_mpiis = []
    visibilities = []

    logger.debug('image process+')
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

    for human in humans:
        pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
        pose_2d_mpiis.append([(int(x*standard_w + 0.5),int(y*standard_h + 0.5)) for x, y in pose_2d_mpii])
        visibilities.append(visibility)

    pose_2d_mpiis = np.array(pose_2d_mpiis)
    visibilities = np.array(visibilities)
    transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpii, visibilities)
    pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)
    return pose_3d

def update_frame():
    ret_val, image = cam.read()
    try:
        keypoints = get_keypoints()
        # Show 3D poses
        for single_3D in keypoints:
            # or plot_pose(Prob3dPose.centre_all(single_3D))
            plot_pose(single_3D)
    except AssertionError:
        print('body not in image')
    else:
        # Show 3D poses
        for single_3D in keypoints:
            # or plot_pose(Prob3dPose.centre_all(single_3D))
            plot_pose(single_3D)

    plt.show()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    logger.info('3d lifting initialization')
    poseLifting = Prob3dPose('LiftingfromtheDeeprelease/data/saved_sessions/prob_model/prob_model_params.mat')

    while True:

        update_frame()
        '''
        ret_val, image = cam.read()

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        
        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        '''
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
    
    '''
    logger.info('3d lifting initialization')
    poseLifting = Prob3dPose('LiftingfromtheDeeprelease/data/saved_sessions/prob_model/prob_model_params.mat')

    image_h, image_w = image.reshape[:2]
    standard_w = 640
    standard_h = 480

    pose_2d_mpiis = []
    visibilities = []
    for human in humans:
        pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
        pose_2d_mpiis.append([(int(x*standard_w + 0.5),int(y*standard_h + 0.5)) for x, y in pose_2d_mpii])
        visibilities.append(visibility)

    pose_2d_mpiis = np.array(pose_2d_mpiis)
    visibilities = np.array(visibilities)
    transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpii, visibilities)
    pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)
    '''