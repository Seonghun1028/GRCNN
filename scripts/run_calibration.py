#!/usr/bin/env python
import numpy as np
import logging

from hardware.calibrate_camera import Calibration

logging.basicConfig(level=logging.INFO) # 추가해준 코드

if __name__ == '__main__':
    calibration = Calibration(
        cam_id=147122072740,
        calib_grid_step=0.05, # 격자 간 간격 
        checkerboard_offset_from_tool=[0.0, 0.0215, 0.0115],
        workspace_limits=np.asarray([[0.55, 0.65], [-0.2, -0.1], [0.0, 0.2]]) # 작업공간 범위(x축 범위, y축 범위, z축 범위)
    )
    calibration.run() # Calibration 클래스 내에서 run 함수 실행
