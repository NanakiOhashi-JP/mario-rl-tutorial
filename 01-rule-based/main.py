import time

import gymnasium as gym
import gym_super_mario_bros

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


env = gym.make(
    "SuperMarioBros-1-1-v0",
    render_mode="human" 
)

env = JoypadSpace(env, SIMPLE_MOVEMENT)

observation, info = env.reset()

former_x_pos = None

for _ in range(1000):
    current_x_pos = info["x_pos"]

    if 270 <= current_x_pos <= 280 or former_x_pos == current_x_pos:
        action = 2
    else:
        action = 1

    former_x_pos = current_x_pos
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(1 / 60)

    if terminated or truncated:
        print(terminated, truncated)
        print(f"Survival Time: {info['time']}")
        input("Press Enter to continue...")
        observation, info = env.reset()
        former_x_pos = None

env.close()
