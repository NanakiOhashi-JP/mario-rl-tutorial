import torch

import gymnasium as gym
import gym_super_mario_bros
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
from nes_py.wrappers import JoypadSpace


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


env = gym.make(
    "SuperMarioBros-1-1-v0",
    render_mode="rgb_array",
)
env = JoypadSpace(env, MOVEMENT)
env = SkipFrame(env, skip=4)

# 画面を小さくし、色の情報を減らす。
env = ResizeObservation(env, (84, 84))
env = GrayscaleObservation(env, keep_dim=False)

# 直近4枚の画面を1つの状態にまとめる。
env = FrameStackObservation(env, stack_size=4)

observation, info = env.reset()

# Tensorへ変換し、ピクセルの値を0〜1にそろえる。
observation = torch.as_tensor(observation, dtype=torch.float32) / 255.0

print(type(observation))
print(observation.shape)
print(observation.dtype)
print(env.action_space)

env.close()
