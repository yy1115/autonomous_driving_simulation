# dqn_agent.py

import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, memory_size=2000):
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
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model = self.build_model(learning_rate)

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
        q_values = self.model.predict(state[np.newaxis, :])
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
        target = self.model.predict(states)
        # 预测下一状态的Q值
        target_next = self.model.predict(next_states)

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

    def load(self, name):
        """
        加载模型权重
        :param name: 模型文件名
        """
        self.model.load_weights(name)

    def save(self, name):
        """
        保存模型权重
        :param name: 模型文件名
        """
        self.model.save_weights(name)
