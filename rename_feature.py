import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---------- 加载旧数据集 ----------
old_dataset = LeRobotDataset(
    repo_id="test1110/merged",
)

# ---------- 修改 features ----------
new_features = old_dataset.features.copy()

# action -> ee_action
if "action" in new_features:
    old_action_feature = new_features.pop("action")
    new_features["ee_action"] = {
        "dtype": old_action_feature["dtype"],
        "shape": old_action_feature["shape"],
        "names": old_action_feature.get("names", [f"{i}" for i in range(old_action_feature["shape"][0])]),
    }

# joint_action -> action
if "joint_action" in new_features:
    joint_action_feature = new_features.pop("joint_action")
    new_features["action"] = {
        "dtype": joint_action_feature["dtype"],
        "shape": joint_action_feature["shape"],
        "names": joint_action_feature.get("names", [f"{i}" for i in range(joint_action_feature["shape"][0])]),
    }

# observation.state -> ee_state
if "observation.state" in new_features:
    state_feature = new_features.pop("observation.state")
    new_features["observation.ee_state"] = {
        "dtype": state_feature["dtype"],
        "shape": state_feature["shape"],
        "names": state_feature.get("names", [f"{i}" for i in range(state_feature["shape"][0])]),
    }

# observation.joint_state -> state
if "observation.joint_state" in new_features:
    joint_state_feature = new_features.pop("observation.joint_state")
    new_features["observation.state"] = {
        "dtype": joint_state_feature["dtype"],
        "shape": joint_state_feature["shape"],
        "names": joint_state_feature.get("names", [f"{i}" for i in range(joint_state_feature["shape"][0])]),
    }

print(new_features)

# ---------- 创建新的空数据集 ----------
FPS = old_dataset.meta.fps
ROBOT_TYPE = old_dataset.meta.robot_type

new_dataset = LeRobotDataset.create(
    repo_id="jajs/merged",
    features=new_features,
    fps=FPS,
    robot_type=ROBOT_TYPE,
    use_videos=True,
    image_writer_threads=4
)

# ---------- 遍历旧数据集，复制特征 ----------
for idx in range(len(old_dataset)):
    sample = old_dataset[idx]
    new_sample = sample.copy()

    # 删除多余索引信息
    for k in ["timestamp", "frame_index", "episode_index", "task_index", "index"]:
        new_sample.pop(k, None)

    # joint_action -> action
    if "joint_action" in sample:
        new_sample["action"] = sample["joint_action"].clone() if isinstance(sample["joint_action"], torch.Tensor) else torch.tensor(sample["joint_action"])

    # action -> ee_action
    if "action" in sample:
        new_sample["ee_action"] = sample["action"].clone() if isinstance(sample["action"], torch.Tensor) else torch.tensor(sample["action"])

    # observation.joint_state -> state
    if "observation.joint_state" in sample:
        new_sample["observation.state"] = sample["observation.joint_state"].clone() if isinstance(sample["observation.joint_state"], torch.Tensor) else torch.tensor(sample["observation.joint_state"])

    # observation.state -> ee_state
    if "observation.state" in sample:
        new_sample["observation.ee_state"] = sample["observation.state"].clone() if isinstance(sample["observation.state"], torch.Tensor) else torch.tensor(sample["observation.state"])

    # 删除旧的 action 和 joint_action
    for k in ["joint_action"]:
        new_sample.pop(k, None)

    # 删除旧的 observation state
    for k in ["observation.joint_state"]:
        if k in new_sample:
            new_sample.pop(k)

    # 处理图像 shape (C,H,W) -> (H,W,C)
    for img_key in ["observation.images.wrist", "observation.images.side"]:
        if img_key in new_sample:
            new_sample[img_key] = new_sample[img_key].permute(1, 2, 0)

    # 添加到新数据集
    new_dataset.add_frame(new_sample)

    # 每1000帧保存一次
    if (idx + 1) % 1000 == 0:
        new_dataset.save_episode()
        new_dataset.episode_buffer = None

# 保存剩余帧
if new_dataset.episode_buffer is not None and new_dataset.episode_buffer["size"] > 0:
    new_dataset.save_episode()
    new_dataset.episode_buffer = None

new_dataset.finalize()

print("新数据集已保存完成！")
