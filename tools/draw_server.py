import re
import matplotlib.pyplot as plt
import numpy as np

# 读取文件
filename = 'server.txt'

# 定义正则表达式模式，用于提取需要的数据
state_joint_obs_pattern = re.compile(r"state_joint_obs:([^\n]+)")
joint_action_pattern = re.compile(r"joint_action=\{([^\}]+)\}")

# 提取出需要的数据
state_joint_obs_data = []
joint_action_data = []
steps = []

with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i in range(0, len(lines), 3):  # 每三个行表示一个数据点
        # 第1行是获取obs的时间, 我们跳过
        # 第2行包含state_joint_obs和joint_action
        line2 = lines[i + 1].strip()

        # 提取state_joint_obs的内容
        state_joint_obs_match = state_joint_obs_pattern.search(line2)
        if state_joint_obs_match:
            state_joint_obs_str = state_joint_obs_match.group(1)
            state_joint_obs = dict(re.findall(r"([\w\.]+)\s*=\s*([\-0-9\.]+)", state_joint_obs_str))
        
        # 提取joint_action的内容
        joint_action_match = joint_action_pattern.search(line2)
        if joint_action_match:
            joint_action_str = joint_action_match.group(1)
            joint_action = dict(re.findall(r"([\w\.]+)\s*=\s*([\-0-9\.]+)", joint_action_str))
        
        # 只取6个关键字的数值
        keys = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 
                'wrist_flex.pos', 'wrist_roll.pos', 'gripper']
        
        # 记录这些数值
        state_vals = [float(state_joint_obs[key]) for key in keys]
        joint_vals = [float(joint_action[key]) for key in keys]

        # 保存数据
        state_joint_obs_data.append(state_vals)
        joint_action_data.append(joint_vals)
        steps.append(i // 3)  # 步骤编号

# 转换为numpy数组便于操作
state_joint_obs_data = np.array(state_joint_obs_data)
joint_action_data = np.array(joint_action_data)

# 绘制图表
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.tight_layout(pad=5.0)

for i, key in enumerate(keys):
    ax = axs[i // 3, i % 3]
    ax.plot(steps, state_joint_obs_data[:, i], label='state_joint_obs', color='b', linestyle='-', marker='o')
    ax.plot(steps, joint_action_data[:, i], label='joint_action', color='r', linestyle='--', marker='x')
    ax.set_title(key)
    ax.set_xlabel('步骤')
    ax.set_ylabel('数值')
    ax.legend()

plt.show()
