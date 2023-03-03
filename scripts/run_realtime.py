import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import save_results, plot_results
from utils.dataset_processing.grasp import detect_grasps                 ##################### 추가 ####################

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default='trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97',
                        help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Connect to Camera
    logging.info('Connecting to camera...')
    cam = RealSenseCamera(device_id=143322072540)
    cam.connect() #카메라 연결
    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb) #다음의 크기의 카메라 데이터로 사용
    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network)
    logging.info('Done')

    # Get the compute device
    device = get_device(args.force_cpu)

    try:
        fig = plt.figure(figsize=(10, 10))
        i = 0
        while i <= 100:   # 100 프레임만 보기
            i += 1
            image_bundle = cam.get_image_bundle() #cam에서 rgb와 depth 이미지를 가져옴
            rgb = image_bundle['rgb']
            depth = image_bundle['aligned_depth']
            x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth) #가져온 데이터를 우리의 카메라 형식으로 전처리
            with torch.no_grad():
                xc = x.to(device)
                pred = net.predict(xc)

                q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
                grasps , grasp_point = detect_grasps(q_img, ang_img, width_img)                              ################### 추가 ##################
                
                transformed_q_img = np.clip(q_img, a_min=0, a_max=1)                                        ################### 추가 ##################
                grasp_score = round(transformed_q_img[grasp_point[0],grasp_point[1]], 2)                    ################### 추가 ##################

                plot_results(fig=fig,
                             rgb_img=cam_data.get_rgb(rgb, False),
                             depth_img=np.squeeze(cam_data.get_depth(depth)),
                             grasp_q_img=q_img,
                             grasp_angle_img=ang_img,
                             score=grasp_score,                                                           ################### 추가 ##################
                             no_grasps=args.n_grasps,
                             grasp_width_img=width_img)
    finally: #중간에 오류가나도 무조건 실행
        save_results(
            rgb_img=cam_data.get_rgb(rgb, False),
            depth_img=np.squeeze(cam_data.get_depth(depth)),
            grasp_q_img=q_img,
            grasp_angle_img=ang_img,
            score=grasp_score,
            no_grasps=args.n_grasps,
            grasp_width_img=width_img
        )


# qt error가 나면 opencv-python 버전을 최소 4.2.0.34 까지 낮춰야 돌아간다. (pip install opencv-python==4.2.0.34)