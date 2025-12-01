import cv2
import numpy as np

# ====== 第一步：计算单应性矩阵 ======
# 读取旧位置图像和新位置图像（第一次采集）
img_old = cv2.imread('old_pos.png')
img_new = cv2.imread('new_pos.png')

# 使用 ORB 检测特征点
orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(img_old, None)
kp2, des2 = orb.detectAndCompute(img_new, None)

# 特征点匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 提取匹配点
src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)  # 新位置
dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)  # 旧位置

# 计算单应性矩阵 H
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# 可以保存 H，方便下次使用
np.save('homography.npy', H)