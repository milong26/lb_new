# 从本地数据集中读取action_ee后ik vs action
import time
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import InverseKinematicsEEToJoints
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.robot_utils import busy_wait

ACTION_EE = "action_ee"
ACTION_JOINT = "action"

EPISODE_IDX = 0
HF_REPO_ID = "complex/joint_ee"

# Initialize the robot config (只是用来获取关节名字，不一定需要连接)
robot_config = SO100FollowerConfig(
    port="/dev/ttyACM0",
    id="congbi",
    use_degrees=True
)
robot = SO100Follower(robot_config)


# Kinematics solver
kinematics_solver = RobotKinematics(
    urdf_path="SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=list(robot.bus.motors.keys()),
)

# Pipeline: EE -> joint
robot_ee_to_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
    steps=[InverseKinematicsEEToJoints(
        kinematics=kinematics_solver,
        motor_names=list(robot.bus.motors.keys()),
        initial_guess_current_joints=False,
    )],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# Load dataset
dataset = LeRobotDataset(HF_REPO_ID, episodes=[EPISODE_IDX])
episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == EPISODE_IDX)

# Extract EE and joint actions
ee_actions = episode_frames.select_columns(ACTION_EE)
joint_actions = episode_frames.select_columns(ACTION_JOINT)
robot.connect()
if not robot.is_connected:
    raise ValueError("Robot is not connected!")
# Store processed joint actions from EE
reconstructed_joints = []

for idx in range(len(episode_frames)):
    t0 = time.perf_counter()
    # EE action
    ee_action = {name: float(ee_actions[idx][ACTION_EE][i]) 
                 for i, name in enumerate(dataset.features[ACTION_EE]["names"])}
    
    robot_obs = robot.get_observation()
    
    # EE -> joint
    joint_action = robot_ee_to_joints_processor((ee_action, robot_obs))
    _ = robot.send_action(joint_action)
    reconstructed_joints.append([joint_action[name] for name in dataset.features[ACTION_JOINT]["names"]])
    busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))


# 原始 joint 数据
original_joints = [
    [float(j) for j in joint_actions[idx][ACTION_JOINT]] 
    for idx in range(len(episode_frames))
]

# Convert to arrays for plotting
import numpy as np
original_joints = np.array(original_joints)
reconstructed_joints = np.array(reconstructed_joints)

# Plot comparison
plt.figure(figsize=(12, 6))
num_joints = original_joints.shape[1]
for j in range(num_joints):
    plt.subplot(num_joints, 1, j+1)
    plt.plot(original_joints[:, j], label="Original Joint")
    plt.plot(reconstructed_joints[:, j], "--", label="EE→Joint Reconstructed")
    plt.ylabel(f"Joint {j}")
    if j == 0:
        plt.title("Original Joint vs EE→Joint Reconstructed")
    if j == num_joints-1:
        plt.xlabel("Frame")
    plt.legend()
plt.tight_layout()
plt.show()
robot.disconnect()
