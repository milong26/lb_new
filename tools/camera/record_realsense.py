# 其他程序没有占用相机的时候运行可以录制到环境

import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 1. 创建 RealSense 管道
    pipeline = rs.pipeline()
    
    # 2. 配置彩色流
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # 3. 启动管道
    pipeline.start(config)
    
    # 4. 视频写入设置
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 编码
    out = cv2.VideoWriter('realsense_output_ee_count.mp4', fourcc, 30.0, (640, 480))
    
    print("按下 'q' 键结束录制")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # 转换为 NumPy 数组
            color_image = np.asanyarray(color_frame.get_data())
            
            # 显示画面
            cv2.imshow('RealSense Color Stream', color_image)
            
            # 写入视频文件
            out.write(color_image)
            
            # 按 q 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # 释放资源
        pipeline.stop()
        out.release()
        cv2.destroyAllWindows()
        print("录制完成，视频已保存为 realsense_output_ee.mp4")

if __name__ == "__main__":
    main()
