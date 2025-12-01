# save as realsense_warp.py
import pyrealsense2 as rs
import numpy as np
import cv2

# ====== 1. 加载之前计算好的单应性矩阵 ======
H = np.load("homography.npy")  # shape (3,3)

# ====== 2. 配置 Realsense 相机 ======
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 30 FPS

# 开启相机
pipeline.start(config)

try:
    while True:
        # 获取帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 转换为 numpy array
        img = np.asanyarray(color_frame.get_data())

        # ====== 显示原始图像 ======
        cv2.imshow("Original", img)

        # ====== 应用单应性矩阵 ======
        h, w = img.shape[:2]
        warped = cv2.warpPerspective(img, H, (w, h))
        cv2.imshow("Warped", warped)

        # 按 q 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
