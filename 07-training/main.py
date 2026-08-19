import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
from nes_py.wrappers import JoypadSpace


TOTAL_STEPS = 1_000_000
MEMORY_CAPACITY = 5000
WARMUP_STEPS = 1000
BATCH_SIZE = 32
TRAIN_INTERVAL = 4
TARGET_UPDATE_INTERVAL = 1000
SAVE_INTERVAL = 10_000

GAMMA = 0.99
LEARNING_RATE = 0.0001
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 50_000

MODEL_PATH = Path(__file__).resolve().parent / "q_network.pt"


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


def make_env(render_mode="rgb_array"):
    env = gym.make(
        "SuperMarioBros-1-1-v0",
        render_mode=render_mode,
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=False)
    env = FrameStackObservation(env, stack_size=4)
    return env


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def states_to_tensor(states, device):
    states = np.stack(states)
    return torch.as_tensor(states, dtype=torch.float32, device=device) / 255.0


def calculate_epsilon(step):
    progress = min(step / EPSILON_DECAY_STEPS, 1.0)
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)


def select_action(state, q_network, env, epsilon, device):
    if random.random() < epsilon:
        return int(env.action_space.sample())

    state_tensor = states_to_tensor([state], device)
    with torch.no_grad():
        q_values = q_network(state_tensor)
    return q_values.argmax(dim=1).item()


def train_step(q_network, target_network, memory, optimizer, device):
    experiences = memory.sample(BATCH_SIZE)

    states = states_to_tensor(
        [experience.state for experience in experiences],
        device,
    )
    actions = torch.tensor(
        [experience.action for experience in experiences],
        dtype=torch.long,
        device=device,
    ).unsqueeze(1)
    rewards = torch.tensor(
        [experience.reward for experience in experiences],
        dtype=torch.float32,
        device=device,
    )
    next_states = states_to_tensor(
        [experience.next_state for experience in experiences],
        device,
    )
    dones = torch.tensor(
        [experience.done for experience in experiences],
        dtype=torch.float32,
        device=device,
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


def main():
    env = make_env()
    device = get_device()
    memory = ReplayMemory(MEMORY_CAPACITY)

    q_network = QNetwork(env.action_space.n).to(device)
    target_network = QNetwork(env.action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = torch.optim.Adam(
        q_network.parameters(),
        lr=LEARNING_RATE,
    )

    state, info = env.reset()
    episode = 1
    episode_reward = 0.0
    latest_loss = None

    print(f"使用デバイス: {device}")

    try:
        for step in range(1, TOTAL_STEPS + 1):
            epsilon = calculate_epsilon(step)
            action = select_action(state, q_network, env, epsilon, device)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            memory.push(
                state=state.copy(),
                action=action,
                reward=float(reward),
                next_state=next_state.copy(),
                done=done,
            )

            episode_reward += reward

            if len(memory) >= WARMUP_STEPS and step % TRAIN_INTERVAL == 0:
                latest_loss = train_step(
                    q_network,
                    target_network,
                    memory,
                    optimizer,
                    device,
                )

            if step % TARGET_UPDATE_INTERVAL == 0:
                target_network.load_state_dict(q_network.state_dict())

            if step % SAVE_INTERVAL == 0:
                torch.save(q_network.state_dict(), MODEL_PATH)
                print(f"モデルを保存しました: {MODEL_PATH}")

            if done:
                loss_text = "--" if latest_loss is None else f"{latest_loss:.4f}"
                print(
                    f"episode={episode} step={step} "
                    f"reward={episode_reward:.1f} "
                    f"epsilon={epsilon:.3f} loss={loss_text}"
                )
                state, info = env.reset()
                episode += 1
                episode_reward = 0.0
            else:
                state = next_state
    except KeyboardInterrupt:
        print("学習を中断しました")
    finally:
        torch.save(q_network.state_dict(), MODEL_PATH)
        env.close()
        print(f"モデルを保存しました: {MODEL_PATH}")


if __name__ == "__main__":
    main()
