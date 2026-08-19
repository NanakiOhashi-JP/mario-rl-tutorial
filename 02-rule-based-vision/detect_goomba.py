import cv2

import time

import gymnasium as gym
import gym_super_mario_bros

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

template = cv2.imread("templates/goomba.png")

env = gym.make(
    "SuperMarioBros-1-1-v0",
    render_mode="human"
)

env = JoypadSpace(env, SIMPLE_MOVEMENT)

observation, info = env.reset()

for _ in range(1000):
    action = 1
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(1 / 60)

    frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)

    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

    _, score, _, position = cv2.minMaxLoc(result)

    if score >= 0.8:
        print(f"クリボー発見！: x = {position[0]}, y = {position[1]}")

    if terminated or truncated:
        print(terminated, truncated)
        print(f"Survival Time: {info['time']}")
        input("Press Enter to continue...")
        observation, info = env.reset()

env.close()