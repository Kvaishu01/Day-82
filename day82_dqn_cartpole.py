# day82_dqn_cartpole.py
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st

# Neural Network for Q-value approximation
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Train DQN Agent
def train_dqn():
    env = gym.make("CartPole-v1", render_mode=None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    q_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    buffer = ReplayBuffer()

    episodes = 100
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(500):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(state_tensor)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, truncated, _ = env.step(action)
            buffer.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if done or truncated:
                break

            if len(buffer.buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states, actions, rewards_b, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards_b = torch.FloatTensor(rewards_b)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0]
                targets = rewards_b + gamma * next_q_values * (1 - dones)

                loss = criterion(q_values, targets.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        target_net.load_state_dict(q_net.state_dict())
        rewards.append(total_reward)
        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")

    env.close()
    return rewards

# Streamlit UI
st.title("🎯 Deep Q-Network – CartPole Balancing")
if st.button("Train DQN Agent"):
    st.info("Training started... please wait ⏳")
    rewards = train_dqn()
    st.success("✅ Training complete!")
    st.line_chart(rewards)
    st.write("Average Reward:", np.mean(rewards))
