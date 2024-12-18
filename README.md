# Autonomous Driving Simulation
![Image text](image.png)
## 项目介绍
本次作业需要为自动驾驶车辆（AV）设计决策和轨迹规划算法，使AV 在多车道交通环境中安全高效地完成换道。考虑到不同的实际交通情况，实验采用强化学习方法实现二次换道甚至多次变道的功能。
## 配置环境
具体环境信息在文件 [requirement.txt](requirement.txt)中，也可直接pip如下
```
pip install numpy matplotlib tensorflow gym
```
## 训练
可直接运行 [main.py](main.py)开始
```
python main.py
```
其中设置了如下参数
```
num_episodes = 500
time_step = 0.1  # 时间步长
highway_length = 1000  # 高速公路长度
num_lanes = 3 # 车道数目
num_vehicles = 6 #车辆总数（包含AV车辆）
av_id = 0  # AV的ID
desired_speed = 25  # AV的期望速度
```
也可自行修改后运行。
其中可视化模块
```
visualizer.plot_highway(highway.vehicles, step)
```
可在训练时注释掉以加快模型训练，在验证时，图片文件将会保存在'output_images'文件夹中，可对输出图片进行转化为gif文件更好展示动图效果

## License
This repository is licensed under MIT licence.

