# Autonomous Driving Simulation
![Image text](image.png)
## 项目介绍
本次作业需要为自动驾驶车辆（AV）设计决策和轨迹规划算法，使AV 在多车道交通环境中安全高效地完成换道。考虑到不同的实际交通情况，实验采用强化学习方法实现二次换道甚至多次变道的功能。
## 配置环境
具体环境信息请查看文件 [requirement.txt](requirement.txt)，也可以直接使用以下命令安装依赖：
```
pip install -r requirements.txt
```
或者使用 pip 安装常用的库：
```
pip install numpy matplotlib tensorflow gym
```
## 训练
可直接运行 [main.py](main.py)开始
```
python main.py
```
在 main.py 中，已设置以下参数：
```
num_episodes = 500
time_step = 0.1  # 时间步长
highway_length = 1000  # 高速公路长度
num_lanes = 3 # 车道数目
num_vehicles = 6 #车辆总数（包含AV车辆）
av_id = 0  # AV的ID
desired_speed = 30  # AV车期望速度
```
也可自行修改后运行。其中HV速度在[highway.py](highway.py)文件中设置为
```
speed = random.uniform(20, 30)  # 随机速度，20-30 m/s
```
并随机分布在高速路前半程路面上，若需要模拟更具体更复杂的道路环境可自行修改实现。

在[main.py](main.py)中可以使用以下代码对训练过程进行可视化：
```
visualizer.plot_highway(highway.vehicles, step)
```
如果你想加快模型的训练过程，可以注释掉该行代码。在验证阶段，生成的图片文件将会保存在 output_images 文件夹中。你也可以将这些图片转换为 GIF 文件，以便更好地展示动态效果。

## License
This repository is licensed under MIT licence.

