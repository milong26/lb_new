# 读取本地数据集
# 提取出action_ee
# 执行

import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
# from lerobot.utils.constants import ACTION
# ACTION="action"
ACTION="action_ee"

from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

EPISODE_IDX = 0
HF_REPO_ID = "test/joint_ee"

# Initialize the robot config
robot_config = SO100FollowerConfig(
    port="/dev/ttyACM0",  # 对应 YAML 的 follower port
    id="congbi",           # YAML 中 follower id
    use_degrees=True
)

# Initialize the robot
robot = SO100Follower(robot_config)

# NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
kinematics_solver = RobotKinematics(
    urdf_path="SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=list(robot.bus.motors.keys()),
)

# Build pipeline to convert EE action to joints action
robot_ee_to_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
    steps=[
        InverseKinematicsEEToJoints(
            kinematics=kinematics_solver,
            motor_names=list(robot.bus.motors.keys()),
            initial_guess_current_joints=False,  # Because replay is open loop
        ),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# Fetch the dataset to replay
dataset = LeRobotDataset(HF_REPO_ID, episodes=[EPISODE_IDX])
# Filter dataset to only include frames from the specified episode since episodes are chunked in dataset V3.0
episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == EPISODE_IDX)
# 不是ACTION，是action_ee!
actions = episode_frames.select_columns(ACTION)
# print("输出",actions)
# Connect to the robot
robot.connect()

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

print("Starting replay loop...")
log_say(f"Replaying episode {EPISODE_IDX}")
for idx in range(len(episode_frames)):
    t0 = time.perf_counter()

    # Get recorded action from dataset
    ee_action = {
        name: float(actions[idx][ACTION][i]) for i, name in enumerate(dataset.features[ACTION]["names"])
    }

    # Get robot observation
    robot_obs = robot.get_observation()

    # Dataset EE -> robot joints
    joint_action = robot_ee_to_joints_processor((ee_action, robot_obs))

    # Send action to robot
    _ = robot.send_action(joint_action)

    busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))

# Clean up
robot.disconnect()
