#!/usr/bin/env python

from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
import time

# --- Robot Configuration ---
robot_config = SO100FollowerConfig(
    port="/dev/ttyACM0",
    id="follower",
    cameras={},  # 不使用相机
    use_degrees=True
)

robot = SO100Follower(robot_config)
robot.connect()

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

# --- Joint Names ---
joint_names = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos"
]

# --- 固定目标关节值 ---
target_action = {
    "shoulder_pan.pos": -10.066,
    "shoulder_lift.pos": -91.433,
    "elbow_flex.pos": 100.533,
    "wrist_flex.pos": 38.086,
    "wrist_roll.pos": -7.041,
    "gripper.pos": 4.487
}

# --- 循环发送直到到位 ---
tolerance = 0.01  # 允许误差 0.05 度
try:
    print("开始移动到目标关节值...")
    while True:
        obs = robot.get_observation()

        # 输出当前关节状态
        print("\n当前关节值：")
        for joint in joint_names:
            print(f"{joint}: {obs.get(joint, 0.0):.3f}")

        # 计算与目标的误差
        diff = [abs(obs[j] - target_action[j]) for j in joint_names]
        if all(d < tolerance for d in diff):
            print("\n所有关节已到达目标位置！")
            break

        # 发送目标动作
        robot.send_action(target_action)

        # 短暂停留，避免过快循环
        time.sleep(0.05)

except KeyboardInterrupt:
    print("程序被用户中断")

finally:
    robot.disconnect()
    print("已断开机器人连接")
