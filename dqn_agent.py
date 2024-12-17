# dqn_agent.py

import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
import os
import json

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, memory_size=2000,
                 model_file='dqn_model.h5', config_file='dqn_config.json'):
        """
        初始化DQN智能体
        :param state_size: 状态空间维度
        :param action_size: 动作空间维度
        :param learning_rate: 学习率
        :param gamma: 折扣因子
        :param epsilon: 探索率
        :param epsilon_decay: 探索率衰减
        :param epsilon_min: 最小探索率
        :param batch_size: 批量大小
        :param memory_size: 经验回放记忆库大小
        :param model_file: 模型文件保存路径
        :param config_file: 配置文件保存路径
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model_file = model_file
        self.config_file = config_file

        # 如果模型文件和配置文件存在，则加载它们
        if os.path.exists(self.model_file) and os.path.exists(self.config_file):
            self.model = tf.keras.models.load_model(self.model_file)
            self.load_config()
            print(f"Loaded model from {self.model_file} and config from {self.config_file}")
        else:
            self.model = self.build_model(learning_rate)
            print("Initialized new model")

    def build_model(self, learning_rate):
        """
        构建DQN模型
        :param learning_rate: 学习率
        :return: 编译后的Keras模型
        """
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        存储经验
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        根据当前状态选择动作（ε-贪心策略）
        :param state: 当前状态
        :return: 动作
        """
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """
        经验回放，训练模型
        """
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # 预测当前状态的Q值
        target = self.model.predict(states, verbose=0)
        # 预测下一状态的Q值
        target_next = self.model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])

        # 训练模型
        self.model.fit(states, target, epochs=1, verbose=0)

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_config(self):
        """
        加载配置文件中的参数
        """
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.epsilon = config.get('epsilon', self.epsilon)
                self.epsilon_decay = config.get('epsilon_decay', self.epsilon_decay)
                self.epsilon_min = config.get('epsilon_min', self.epsilon_min)
            print(f"Config loaded from {self.config_file}")
        else:
            print(f"No config file found at {self.config_file}")

    def save_config(self):
        """
        保存当前的配置参数到文件
        """
        config = {
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
        print(f"Config saved to {self.config_file}")

    def save(self):
        """
        保存模型和配置
        """
        self.model.save(self.model_file)
        self.save_config()
        print(f"Model and config saved to {self.model_file} and {self.config_file}")

    def load(self):
        """
        加载模型和配置
        """
        if os.path.exists(self.model_file) and os.path.exists(self.config_file):
            self.model = tf.keras.models.load_model(self.model_file)
            self.load_config()
            print(f"Model and config loaded from {self.model_file} and {self.config_file}")
        else:
            print(f"No model or config file found at {self.model_file} and {self.config_file}")
