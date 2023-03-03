#! /usr/bin/env python

##### import os         #####
##### import argparse   #####
##### import logging    #####
##### import tf         #####

import time
import rospy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data

from math import pi
from grcnn.srv import GraspPrediction, GraspPredictionResponse
from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps
from utils.visualisation.plot import save_results, plot_results
from tf.transformations import *



class GRCNNService:
    def __init__(self):
        self.saved_model_path = 'trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97'
        self.camera = RealSenseCamera(device_id=143322071682) # 143322072540 
        self.model = torch.load(self.saved_model_path) # load_model에서 불러옴
        self.device = get_device(force_cpu=False) # load_model에서 불러옴
        self.cam_data = CameraData(include_depth=True, include_rgb=True)
        self.pipe = self.camera.connect() # Connect to camera
        self.cam_depth_scale = 0.001 #scale은 1

        rospy.Service('/predict', GraspPrediction, self.compute_service_handler) 


    def compute_service_handler(self, req):
        try:

            fig = plt.figure(figsize=(10, 10))

            i = 0
            while i < 30:
                i += 1
                
                image_bundle = self.camera.get_image_bundle() #cam에서 rgb와 depth 이미지를 가져옴
                rgb = image_bundle['rgb']
                depth = image_bundle['aligned_depth']
                raw_depth = image_bundle['raw_depth']                                                                        ############ 추가 ############
                x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth) #가져온 데이터를 우리의 카메라 형식으로 전처리

                with torch.no_grad():

                    xc = x.to(self.device)
                    pred = self.model.predict(xc)
                    q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
                    grasps , grasp_point = detect_grasps(q_img, ang_img, width_img)                                          ############ 추가 ############
                    transformed_q_img = np.clip(q_img, a_min=0, a_max=1)                                        ################### 추가 ##################
                    grasp_score = round(transformed_q_img[grasp_point[0],grasp_point[1]], 2)                    ################### 추가 ##################

                    plot_results(fig=fig,
                                    rgb_img=self.cam_data.get_rgb(rgb, False),
                                    depth_img=np.squeeze(self.cam_data.get_depth(depth)),
                                    grasp_q_img=q_img,
                                    grasp_angle_img=ang_img,
                                    score=grasp_score,                                                             ############ 추가 ############
                                    no_grasps=1,
                                    grasp_width_img=width_img)
        finally:
            self.pipe.stop()
        #     save_results(
        #     rgb_img=cam_data.get_rgb(rgb, False),
        #     depth_img=np.squeeze(cam_data.get_depth(depth)),
        #     grasp_q_img=q_img,
        #     grasp_angle_img=ang_img,
        #     score=grasp_score,
        #     no_grasps=args.n_grasps,
        #     grasp_width_img=width_img
        # )

        time.sleep(3)

        grasps , grasp_point = detect_grasps(q_img, ang_img, width_img) #파지점 추출                        ######### grasp_point 추가 #########

        # pos_z = depth[grasps[0].center[0] + self.cam_data.top_left[0], grasps[0].center[1] + self.cam_data.top_left[1]] * self.cam_depth_scale
        pos_z = raw_depth[grasps[0].center[0] + self.cam_data.top_left[0], grasps[0].center[1] + self.cam_data.top_left[1]] * self.cam_depth_scale  ###### 수정 #####
        pos_x = np.multiply(grasps[0].center[1] + self.cam_data.top_left[1] - self.camera.intrinsics.ppx,
                            pos_z / self.camera.intrinsics.fx)
        pos_y = np.multiply(grasps[0].center[0] + self.cam_data.top_left[0] - self.camera.intrinsics.ppy,
                            pos_z / self.camera.intrinsics.fy)

        if pos_z == 0:
            print("Failed to get correct value from depth")

        # Camera to target distance
        target = np.asarray([pos_x, pos_y, pos_z])
        target.shape = (3, 1)
        print('Camera to target: ', target)

        # Convert camera to robot coordinates
        target_position = [(0.30713270+0.045)-pos_y, (0.03)-pos_x,(0.58663631+0.06)-pos_z]
        print('Base to target: ', target_position)

        # Convert camera to robot angle
        print('Grasp angle: ', grasps[0].angle)

        target_angle = [pi, 0, pi/4+grasps[0].angle]
        print('Orientation in Euler: ', target_angle)

        ret = GraspPredictionResponse() # bool sussess 
        ret.success = True
        g = ret.best_grasp # 

        g.position.x = target_position[0]
        g.position.y = target_position[1]
        g.position.z = target_position[2]

        quat = quaternion_from_euler(target_angle[0],target_angle[1],target_angle[2])
        g.orientation.x = quat[0]
        g.orientation.y = quat[1]
        if g.orientation.y >= 0:
            g.orientation.y *= -1
        g.orientation.z = quat[2]
        g.orientation.w = quat[3]

        return ret

if __name__ == '__main__':
    rospy.init_node('grcnn_service')
    GRCNN = GRCNNService()
    rospy.spin()