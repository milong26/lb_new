import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---------- 加载旧数据集 ----------
# old_dataset_path = "/home/qwe/.cache/huggingface/test1110/merged"
old_dataset = LeRobotDataset(
    repo_id="test1110/merged",
    # root=old_dataset_path,
)

# ---------- 修改features ----------
new_features = old_dataset.features.copy()

# ---------- 修改features ----------
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

print(new_features)
# ---------- 创建新的空数据集 ----------
# new_dataset_path = "/home/qwe/.cache/huggingface/test1111/merged"
FPS = old_dataset.meta.fps
ROBOT_TYPE = old_dataset.meta.robot_type

new_dataset = LeRobotDataset.create(
    repo_id="test1111/merged",
    # root=new_dataset_path,
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


    for k in ["timestamp", "frame_index", "episode_index", "task_index", "index"]:
        if k in new_sample:
            new_sample.pop(k)

    if "joint_action" in sample:
        new_sample["action"] = sample["joint_action"].clone() if isinstance(sample["joint_action"], torch.Tensor) else torch.tensor(sample["joint_action"])

    # ee_action 从 action 复制
    if "action" in sample:
        new_sample["ee_action"] = sample["action"].clone() if isinstance(sample["action"], torch.Tensor) else torch.tensor(sample["action"])

    # 删除原来的 action 和 joint_action
    for k in ["joint_action"]:
        if k in new_sample:
            new_sample.pop(k)

    # 处理图像 shape (C,H,W) -> (H,W,C)
    for img_key in ["observation.images.wrist", "observation.images.side"]:
        if img_key in new_sample:
            new_sample[img_key] = new_sample[img_key].permute(1, 2, 0)

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
