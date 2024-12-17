# vehicle.py

import numpy as np

class Vehicle:
    def __init__(self, vehicle_id, lane, position, speed, is_av=False):
        """
        初始化车辆对象
        :param vehicle_id: 车辆唯一标识
        :param lane: 当前所在车道（0, 1, 2）
        :param position: 车辆当前位置（距离起点的距离）
        :param speed: 车辆当前速度（单位：m/s）
        :param is_av: 是否为自动驾驶车辆（AV）
        """
        self.vehicle_id = vehicle_id
        self.lane = lane
        self.position = position
        self.speed = speed
        self.is_av = is_av
        self.target_lane = lane  # 目标车道，用于换道

    def update_position(self, time_step=0.1):
        """
        更新车辆位置
        :param time_step: 时间步长
        """
        self.position += self.speed * time_step
