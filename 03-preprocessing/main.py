import torch

import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
from nes_py.wrappers import JoypadSpace

env = gym.make(
    "SuperMarioBros-1-1-v0",
    render_mode="rgb_array",
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

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
