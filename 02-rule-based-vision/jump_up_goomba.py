import time

import cv2
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

template = cv2.imread("templates/goomba.png")

if template is None:
    raise FileNotFoundError("クリボーの画像が見つかりません")

env = gym.make(
    "SuperMarioBros-1-1-v0",
    render_mode="human"
)

env = JoypadSpace(env, SIMPLE_MOVEMENT)

observation, info = env.reset()

goomba_x = None

for _ in range(1000):
    mario_x = info["left_x_pos"]

    if goomba_x is not None and 20 < goomba_x - mario_x < 50:
        action = 2
    else:
        action = 1

    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(1 / 60)

    frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)

    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

    _, score, _, position = cv2.minMaxLoc(result)

    if score >= 0.8:
        goomba_x = position[0]
        print(f"クリボー発見！: x = {goomba_x}, score = {score:.3f}")
    else:
        goomba_x = None

    if terminated or truncated:
        input("Press Enter to continue...")
        observation, info = env.reset()
        goomba_x = None

env.close()