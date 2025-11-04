import torch
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say

# 加载数据集
local_dataset_path = "/home/qwe/.cache/huggingface/lerobot/test/joint"
dataset = LeRobotDataset(
    repo_id="test/joint",
    root=local_dataset_path,
)

# ---------- 创建Kinematics 正向运动学 ----------
joint_names = [name.replace(".pos", "") for name in dataset.features["action"]["names"]]
urdf_path = "SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
kinematics_solver = RobotKinematics(
    urdf_path=urdf_path,
    target_frame_name="gripper_frame_link",
    joint_names=joint_names
)

# Build pipeline to convert follower joints to EE observation
follower_joints_to_ee = RobotProcessorPipeline[RobotObservation, RobotObservation](
    steps=[
        ForwardKinematicsJointsToEE(
            kinematics=kinematics_solver, 
            # 得到motor的名字
            motor_names=joint_names
        ),
    ],
    to_transition=observation_to_transition, # 把输入（RobotObservation）转换为 pipeline 内部使用的统一格式
    to_output=transition_to_observation, # pipeline 内部处理后的统一格式转换回输出格式
)

# Build pipeline to convert leader joints to EE action
leader_joints_to_ee = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
    steps=[
        ForwardKinematicsJointsToEE(
            kinematics=kinematics_solver, motor_names=joint_names
        ),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)


# ---------- 在features里添加EE字段 ----------
new_features = dataset.features.copy()

# action_ee
new_features["action_ee"] = {
    "dtype": "float32",
    "shape": (7,),
    "names": ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"],
}

# observation.state_ee
new_features["observation.state_ee"] = {
    "dtype": "float32",
    "shape": (7,),
    "names": ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"],
}

# ---------- 创建新的空数据集 ----------
new_dataset_path = "/home/qwe/.cache/huggingface/lerobot/test/joint_ee"
FPS = dataset.meta.fps
ROBOT_TYPE = dataset.meta.robot_type

new_dataset = LeRobotDataset.create(
    repo_id="test/joint_ee",
    root=new_dataset_path,
    features=new_features,
    fps=FPS,
    robot_type=ROBOT_TYPE,
    use_videos=True,
    image_writer_threads=4
)


# ---------- 遍历原始数据集，计算EE并保存 ----------
for idx in range(len(dataset)):
    sample = dataset[idx]
    enhanced_sample = sample.copy()
    action_dict = {name: sample["action"][i].item() 
                   for i, name in enumerate(dataset.features["action"]["names"])}
    obs_dict = {name: sample["observation.state"][i].item() 
                for i, name in enumerate(dataset.features["observation.state"]["names"])}
    action_ee = leader_joints_to_ee((action_dict, obs_dict))  # 只传一个tuple
    ee_values = [action_ee[n] for n in new_features["action_ee"]["names"]]
    enhanced_sample["action_ee"] = torch.tensor(ee_values, dtype=torch.float32)

    # 计算observation.state_ee
    obs_ee = follower_joints_to_ee(obs_dict)  # 只传一个参数
    ee_values = [obs_ee.get(n, 0.0) for n in new_features["observation.state_ee"]["names"]]
    enhanced_sample["observation.state_ee"] = torch.tensor(ee_values, dtype=torch.float32)
    # ---------- 写入新dataset ----------
    # 去掉多余的字段，lerobotdatse会自动生成
    for k in ["timestamp", "frame_index", "episode_index", "task_index", "index"]:
        if k in enhanced_sample:
            enhanced_sample.pop(k)

    # 为什么要先处理图像The feature 'observation.images.wrist' of shape '(3, 480, 640)' does not have the expected shape '(480, 640, 3)' or '(640, 3, 480)'.
    if "observation.images.wrist" in enhanced_sample:
        # (C, H, W) -> (H, W, C)
        enhanced_sample["observation.images.wrist"] = enhanced_sample["observation.images.wrist"].permute(1, 2, 0)

    new_dataset.add_frame(enhanced_sample)

    # 每1000帧写一次episode，避免内存过大
    if (idx + 1) % 1000 == 0:
        new_dataset.save_episode()
        new_dataset.episode_buffer = None  # 重置episode_buffer

# 保存剩余帧
if new_dataset.episode_buffer is not None and new_dataset.episode_buffer["size"] > 0:
    new_dataset.save_episode()
    new_dataset.episode_buffer = None

new_dataset.finalize()

print("增强数据集已保存完成！")