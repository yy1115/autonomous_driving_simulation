# main.py

import numpy as np
from highway import Highway
from dqn_agent import DQNAgent
from trajectory_planner import TrajectoryPlanner
from visualization import Visualization
import os

def main():
    # 初始化参数
    num_episodes = 500
    time_step = 0.1  # 时间步长
    highway_length = 1000  # 高速公路长度
    num_lanes = 3
    num_vehicles = 6
    av_id = 0  # AV的ID
    desired_speed = 25  # AV的期望速度
    model_file = 'dqn_model.h5'
    config_file = 'dqn_config.json'

    # 初始化环境、智能体、轨迹规划和可视化
    highway = Highway(num_lanes=num_lanes, num_vehicles=num_vehicles, av_id=av_id, highway_length=highway_length, desired_speed=desired_speed)
    state_size = 4  # [车道, 距离, 速度, 前车速度]
    action_size = 3  # [保持, 左换道, 右换道]
    agent = DQNAgent(state_size=state_size, action_size=action_size, model_file=model_file, config_file=config_file)
    planner = TrajectoryPlanner()
    visualizer = Visualization(highway_length=highway_length, num_lanes=num_lanes)

    # 检查是否有已保存的模型，如果有则加载
    if os.path.exists(model_file) and os.path.exists(config_file):
        agent.load()
    else:
        print("No existing model found, starting training from scratch.")

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

        print(f"Episode {episode+1}/{num_episodes} - Steps: {step} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")

        # 每1个episode保存一次模型
        if (episode + 1) % 1 == 0:
            agent.save()
            print(f"Model saved at episode {episode+1}")

    print("训练完成。")

if __name__ == "__main__":
    main()
