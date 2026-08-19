import random
from collections import deque
from dataclasses import dataclass

import numpy as np

import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
from nes_py.wrappers import JoypadSpace


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


def make_env():
    env = gym.make(
        "SuperMarioBros-1-1-v0",
        render_mode="rgb_array",
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=False)
    env = FrameStackObservation(env, stack_size=4)
    return env


def main():
    env = make_env()
    memory = ReplayMemory(capacity=1000)
    state, info = env.reset()

    for _ in range(500):
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

    experiences = memory.sample(batch_size=4)

    print(f"保存した経験: {len(memory)}個")
    print(f"取り出した経験: {len(experiences)}個")
    print(f"状態の形: {experiences[0].state.shape}")
    print(f"行動: {experiences[0].action}")
    print(f"報酬: {experiences[0].reward}")
    print(f"終了したか: {experiences[0].done}")

    env.close()


if __name__ == "__main__":
    main()
