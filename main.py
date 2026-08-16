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

survival_time = 0

for _ in range(1000):

    if survival_time == 150:
        action = 2
    else:
        action = 1

    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(1 / 60)
    survival_time += 1


    if terminated or truncated:
        print(terminated, truncated)
        print(f"Survival Time: {survival_time}")
        input("Press Enter to continue...")
        observation, info = env.reset()
        survival_time = 0

env.close()