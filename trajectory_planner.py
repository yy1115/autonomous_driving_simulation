# trajectory_planner.py

import numpy as np

class TrajectoryPlanner:
    def __init__(self):
        pass

    def plan_trajectory(self, current_position, target_position, current_speed, time_step=0.1):
        """
        简单的直线轨迹规划
        :param current_position: 当前位置信息
        :param target_position: 目标位置信息
        :param current_speed: 当前速度
        :param time_step: 时间步长
        :return: 轨迹点列表
        """
        trajectory = []
        while current_position < target_position:
            current_position += current_speed * time_step
            trajectory.append(current_position)
        return trajectory
