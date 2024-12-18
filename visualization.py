import matplotlib.pyplot as plt
import os

class Visualization:
    def __init__(self, highway_length=1000, num_lanes=3, save_dir="output_images"):
        self.highway_length = highway_length
        self.num_lanes = num_lanes
        self.fig, self.ax = plt.subplots()
        
        # 创建保存图像的目录（如果不存在的话）
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def plot_highway(self, vehicles, step):
        """
        绘制当前高速公路状态，并保存输出图片
        :param vehicles: 当前所有车辆
        :param step: 当前时间步
        """
        self.ax.clear()
        lanes = self.num_lanes
        highway_length = self.highway_length

        # 绘制车道线
        for lane in range(lanes + 1):
            y = lane * 3.5
            self.ax.plot([0, highway_length], [y, y], 'k--')

        # 绘制车辆
        for vehicle in vehicles:
            lane_center = vehicle.lane * 3.5 + 1.75  # 车道中心线
            if vehicle.is_av:
                self.ax.plot(vehicle.position, lane_center, 'ro', label='AV')
            else:
                self.ax.plot(vehicle.position, lane_center, 'bo', label='HV')

        self.ax.set_xlim(0, highway_length)
        self.ax.set_ylim(-1, lanes * 3.5 + 1)
        self.ax.set_xlabel('位置 (米)')
        self.ax.set_ylabel('车道')
        self.ax.set_title(f'高速公路仿真 - 时间步 {step}')
        self.ax.legend(['车道线', 'AV', 'HV'], loc='upper right')

        # 保存图片
        save_path = os.path.join(self.save_dir, f"highway_step_{step}.png")
        self.fig.savefig(save_path)

        # 显示图像
        plt.pause(0.001)
