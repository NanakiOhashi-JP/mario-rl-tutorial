import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

import gymnasium as gym
import gym_super_mario_bros
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
from nes_py.wrappers import JoypadSpace


BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.0001

MOVEMENT = [
    ["right"],
    ["right", "A"],
]


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0

        for _ in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return observation, total_reward, terminated, truncated, info


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def make_env():
    env = gym.make(
        "SuperMarioBros-1-1-v0",
        render_mode="rgb_array",
    )
    env = JoypadSpace(env, MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=False)
    env = FrameStackObservation(env, stack_size=4)
    return env


def states_to_tensor(states):
    states = np.stack(states)
    return torch.as_tensor(states, dtype=torch.float32) / 255.0


def train_step(q_network, target_network, memory, optimizer):
    experiences = memory.sample(BATCH_SIZE)

    states = states_to_tensor([experience.state for experience in experiences])
    actions = torch.tensor(
        [experience.action for experience in experiences],
        dtype=torch.long,
    ).unsqueeze(1)
    rewards = torch.tensor(
        [experience.reward for experience in experiences],
        dtype=torch.float32,
    )
    next_states = states_to_tensor(
        [experience.next_state for experience in experiences]
    )
    dones = torch.tensor(
        [experience.done for experience in experiences],
        dtype=torch.float32,
    )

    predicted_q_values = q_network(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q_values = target_network(next_states).max(dim=1).values
        target_q_values = rewards + GAMMA * next_q_values * (1.0 - dones)

    loss = nn.SmoothL1Loss()(predicted_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def collect_experiences(env, memory, count):
    state, info = env.reset()

    for _ in range(count):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        memory.push(
            state=state.copy(),
            action=int(action),
            reward=float(reward),
            next_state=next_state.copy(),
            done=done,
        )

        if done:
            state, info = env.reset()
        else:
            state = next_state


def main():
    env = make_env()
    memory = ReplayMemory(capacity=1000)

    q_network = QNetwork(n_actions=env.action_space.n)
    target_network = QNetwork(n_actions=env.action_space.n)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = torch.optim.Adam(
        q_network.parameters(),
        lr=LEARNING_RATE,
    )

    collect_experiences(env, memory, count=500)
    loss = train_step(q_network, target_network, memory, optimizer)

    print(f"保存した経験: {len(memory)}個")
    print(f"バッチサイズ: {BATCH_SIZE}")
    print(f"損失: {loss:.4f}")

    env.close()


if __name__ == "__main__":
    main()
