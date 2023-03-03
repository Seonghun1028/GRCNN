import logging

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


class RealSenseCamera:
    def __init__(self,
                 device_id,
                 width=640,
                 height=480,
                 fps=15):                                                                                        ########### 6 -> 15 변경 ############
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None # pipeline : 한 데이터 처리 단계의 출력이 다음 단계의 입력으로 이어지는 형태로 연결된 구조
        self.scale = None
        self.intrinsics = None

    def connect(self):
        # Start and configure (시작 및 환경 설정)
        self.pipeline = rs.pipeline() # realsense pipeline open
        config = rs.config()
        config.enable_device(str(self.device_id))
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        cfg = self.pipeline.start(config)

        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()  
        # [ 640x480  p[328.434 234.973]  f[603.122 603.203]  Inverse Brown Conrady [0 0 0 0 0] ]
        # [ Width x Height p[PPX PPY] f[Fx Fy] Distortion [Coeffs]]

        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale() 
        # 0.0010000000474974513
        return self.pipeline                                                                                                 ########## 추가 ##########

    def get_image_bundle(self):
        frames = self.pipeline.wait_for_frames() #color와 depth의 프레임셋을 기다림

        raw_depth_image = np.asarray(frames.get_depth_frame().get_data(), dtype=np.float32)                                 ########## 추가 ##########

        align = rs.align(rs.stream.color) # depth 이미지와 맞추기 위한 align 생성
        aligned_frames = align.process(frames) # 모든(depth 포함) 프레임을 컬러 프레임에 맞추어 변환
        color_frame = aligned_frames.first(rs.stream.color) # color 프레임 얻음
        aligned_depth_frame = aligned_frames.get_depth_frame() # depth 프레임 얻음

        depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32) # depth 이미지를 배열로 (480,640)
        depth_image *= self.scale # 0.001 곱해줌
        color_image = np.asanyarray(color_frame.get_data()) # color 이미지를 배열로 (480,640,3)

        depth_image = np.expand_dims(depth_image, axis=2) # depth 이미지의 차원 늘리기 (480,640) -> (480,640,1)

        return {
            'rgb': color_image,
            'aligned_depth': depth_image,
            'raw_depth' : raw_depth_image                                                                                  ########## 추가 ##########
        }

    def plot_image_bundle(self):
        images = self.get_image_bundle() # get_image_bundle() 함수로부터 color,depth image를 배열형태로 받음

        rgb = images['rgb']
        depth = images['aligned_depth']

        fig, ax = plt.subplots(1, 2, squeeze=False)
        ax[0, 0].imshow(rgb)
        m, s = np.nanmean(depth), np.nanstd(depth) # nanmean, nanstd : NaN값을 무시하고 평균, 표준편차 계산
        ax[0, 1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap=plt.cm.gray) # normalization 
        ax[0, 0].set_title('rgb')
        ax[0, 1].set_title('aligned_depth')

        plt.show()


if __name__ == '__main__':
    cam = RealSenseCamera(device_id=143322072540)
    cam.connect()
    while True:
        cam.plot_image_bundle()
        