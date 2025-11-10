import pybullet as p
import pybullet_data
import time

# 连接 PyBullet GUI
physicsClient = p.connect(p.GUI)

# 设置模型搜索路径
p.setAdditionalSearchPath("SO-ARM100/Simulation/SO101")

# 加载 URDF
robot = p.loadURDF("so101_new_calib.urdf", useFixedBase=True)

# 获取关节数量
num_joints = p.getNumJoints(robot)
print("关节数:", num_joints)

# 打印关节信息并设置到零位
for j in range(num_joints):
    info = p.getJointInfo(robot, j)
    name = info[1].decode('utf-8')
    joint_type = info[2]
    lower, upper = info[8], info[9]
    print(f"{j}: {name}, lower={lower}, upper={upper}, type={joint_type}")

    # 如果是可转动关节（revolute/prismatic），设置到零位
    if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        p.resetJointState(robot, j, targetValue=0.0)

# 🔹 所有关节都归零后暂停观察
print("所有关节已设置到 0 位，开始显示。")
time.sleep(10)  # 暂停 10 秒（你可以改成更久，比如 60）

# （可选）关闭物理仿真保持静止
p.setRealTimeSimulation(0)

# 保持窗口开启直到你手动关闭
while True:
    time.sleep(0.1)
