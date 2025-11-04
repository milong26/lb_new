原来的md太长了，见github.com/huggingface/lerobot

去掉了tests和大文件

1. 增加so101的urdf文件，转换本地原来dataset到一个新的，需要4.0的datset，并且比较局限，只能做so100，没验证。修改new_change.py里面的配置。新的叫action_ee和state_ee，原来的是action和state
- [ ] 是否需要重构成action.joint.xxx和action.ee.xxx这样的形式？
2. 可视化数据集：python src/lerobot/scripts/lerobot_dataset_viz.py --repo-id test/joint_ee --episode-index 0这样执行，如果直接用脚本的话不行
3. replay：用ee的树脂replay，需要反解。python examples/so100_to_so100_EE/new_replay.py 
4. compre：比较真实action和ik解出的action，绘制图片