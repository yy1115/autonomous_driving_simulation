# main.py

import numpy as np
from highway import Highway
from dqn_agent import DQNAgent
from trajectory_planner import TrajectoryPlanner
from visualization import Visualization

def main():
    # 初始化参数
    num_episodes = 500
    time_step = 0.1  # 时间步长
    highway_length = 1000  # 高速公路长度
    num_lanes = 3
    num_vehicles = 6
    av_id = 0  # AV的ID

    # 初始化环境、智能体、轨迹规划和可视化
    highway = Highway(num_lanes=num_lanes, num_vehicles=num_vehicles, av_id=av_id, highway_length=highway_length)
    state_size = 4  # [车道, 距离, 速度, 前车速度]
    action_size = 3  # [保持, 左换道, 右换道]
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    planner = TrajectoryPlanner()
    visualizer = Visualization(highway_length=highway_length, num_lanes=num_lanes)

    for episode in range(num_episodes):
        state = highway.reset()
        done = False
        step = 0
        total_reward = 0

        while not done and step < 1000:
            action = agent.act(state)
            next_state, reward, done = highway.step(action, time_step)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1

            # 可视化
            visualizer.plot_highway(highway.vehicles, step)

            # 训练智能体
            agent.replay()

        print(f"Episode {episode+1}/{num_episodes} - Steps: {step} - Total Reward: {total_reward} - Epsilon: {agent.epsilon:.2f}")

        # 每10个episode保存一次模型
        if (episode + 1) % 10 == 0:
            agent.save(f'dqn_model_episode_{episode+1}.h5')

    print("训练完成。")

if __name__ == "__main__":
    main()
