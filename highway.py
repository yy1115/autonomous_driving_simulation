# highway.py

import random
from vehicle import Vehicle

class Highway:
    def __init__(self, num_lanes=3, num_vehicles=6, av_id=0, highway_length=1000):
        """
        初始化高速公路环境
        :param num_lanes: 车道数量
        :param num_vehicles: 总车辆数量（包括AV和HV）
        :param av_id: AV的车辆ID
        :param highway_length: 高速公路长度（单位：米）
        """
        self.num_lanes = num_lanes
        self.num_vehicles = num_vehicles
        self.highway_length = highway_length
        self.vehicles = []
        self.av_id = av_id
        self.initialize_vehicles()

    def initialize_vehicles(self):
        """
        创建车辆（1辆AV和5辆HV）
        """
        for i in range(self.num_vehicles):
            lane = random.randint(0, self.num_lanes - 1)
            position = random.uniform(0, self.highway_length / 2)  # 初始位置在高速公路前半段
            speed = random.uniform(20, 30)  # 随机速度，20-30 m/s
            is_av = (i == self.av_id)
            vehicle = Vehicle(vehicle_id=i, lane=lane, position=position, speed=speed, is_av=is_av)
            self.vehicles.append(vehicle)

    def get_av_vehicle(self):
        """
        获取AV车辆对象
        :return: AV车辆对象
        """
        return self.vehicles[self.av_id]

    def get_vehicle_ahead(self, av_vehicle):
        """
        获取AV前方最近的车辆
        :param av_vehicle: AV车辆对象
        :return: 前方车辆对象或None
        """
        vehicles_ahead = [v for v in self.vehicles if v.lane == av_vehicle.lane and v.position > av_vehicle.position]
        if not vehicles_ahead:
            return None
        return min(vehicles_ahead, key=lambda v: v.position)

    def get_state(self):
        """
        获取当前状态
        状态包括：
        - AV所在车道
        - AV与前车的距离
        - AV速度
        - 前车速度
        :return: 状态向量
        """
        av = self.get_av_vehicle()
        front_vehicle = self.get_vehicle_ahead(av)
        distance = front_vehicle.position - av.position if front_vehicle else self.highway_length
        front_speed = front_vehicle.speed if front_vehicle else 0
        state = np.array([av.lane, distance, av.speed, front_speed], dtype=np.float32)
        return state

    def step(self, action, time_step=0.1):
        """
        执行一步仿真
        :param action: 动作（0: 保持车道, 1: 向左换道, 2: 向右换道）
        :param time_step: 时间步长
        :return: 下一状态, 奖励, 是否结束
        """
        av = self.get_av_vehicle()

        # 执行动作
        if action == 1 and av.lane > 0:
            av.target_lane = av.lane - 1
        elif action == 2 and av.lane < self.num_lanes - 1:
            av.target_lane = av.lane + 1
        else:
            av.target_lane = av.lane  # 保持车道

        # 简单的换道实现：目标车道与当前车道不同，则换道
        if av.target_lane != av.lane:
            av.lane = av.target_lane  # 实际应用中应平滑过渡

        # 更新所有车辆的位置
        for vehicle in self.vehicles:
            vehicle.update_position(time_step)

        # 获取下一状态
        next_state = self.get_state()

        # 计算奖励
        reward = self.compute_reward()

        # 判断是否结束（例如，AV到达高速公路尽头）
        done = av.position >= self.highway_length

        return next_state, reward, done

    def compute_reward(self):
        """
        计算奖励函数
        奖励设计：
        - 保持与前车的安全距离时给予正奖励
        - 碰撞时给予负奖励
        - 换道成功时给予奖励
        - 不必要的换道时给予惩罚
        - 到达目的地时给予奖励
        :return: 奖励值
        """
        av = self.get_av_vehicle()
        front_vehicle = self.get_vehicle_ahead(av)
    
        # 碰撞检测：与前车距离小于5米
        if front_vehicle:
            distance = front_vehicle.position - av.position
            if distance < 5:
                return -100  # 碰撞惩罚
    
        # 距离过近：与前车距离小于20米
        if front_vehicle and distance < 20:
            return -1  # 接近前车，避免碰撞的惩罚
    
        # 保持安全距离：距离前车大于20米
        if front_vehicle and distance >= 20:
            return 1  # 安全驾驶，保持足够的距离
    
        # 前方无车：给较大的正奖励
        if not front_vehicle:
            return 10  # 前方没有车辆，AV可以自由行驶
    
        # 换道奖励：只有在成功换道且有利于行车时，给予奖励
        if av.target_lane != av.lane:
            # 如果换道是为了避免前车
            if front_vehicle and distance < 20:
                return 5  # 通过换道避免了前车的危险
            else:
                return -2  # 如果换道不必要，惩罚
    
        # 到达目的地奖励：如果AV到达了高速公路尽头，给予奖励
        if av.position >= self.highway_length:
            return 100  # 到达目的地时给予奖励
    
        # 默认奖励：如果没有特别情况，则保持当前行为
        return 0

    def reset(self):
        """
        重置高速公路环境
        """
        self.vehicles = []
        self.initialize_vehicles()
        return self.get_state()
