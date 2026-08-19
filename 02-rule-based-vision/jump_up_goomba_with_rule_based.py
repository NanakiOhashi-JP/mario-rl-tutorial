import time
from pathlib import Path

import cv2
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "goomba.png"
MATCH_THRESHOLD = 0.8
MIN_GOOMBA_DISTANCE = 20
MAX_GOOMBA_DISTANCE = 50
STUCK_FRAME_THRESHOLD = 10

template = cv2.imread(str(TEMPLATE_PATH))
if template is None:
    raise FileNotFoundError(f"クリボーの画像が見つかりません: {TEMPLATE_PATH}")

env = gym.make(
    "SuperMarioBros-1-1-v0",
    render_mode="human",
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

observation, info = env.reset()

goomba_x = None
former_x_pos = None
stuck_frame_count = 0

try:
    for _ in range(1000):
        current_x_pos = info["x_pos"]

        # 同じステージ座標から動けないフレーム数を数える。
        if former_x_pos is not None and former_x_pos == current_x_pos:
            stuck_frame_count += 1
        else:
            stuck_frame_count = 0

        mario_x = info["left_x_pos"]
        goomba_distance = None if goomba_x is None else goomba_x - mario_x

        is_goomba_nearby = (
            goomba_distance is not None
            and MIN_GOOMBA_DISTANCE < goomba_distance < MAX_GOOMBA_DISTANCE
        )
        is_stuck = stuck_frame_count >= STUCK_FRAME_THRESHOLD

        # クリボーが近い、または10フレーム連続で動けないときにジャンプする。
        action = 2 if is_goomba_nearby or is_stuck else 1

        former_x_pos = current_x_pos
        observation, _, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(1 / 60)

        # テンプレートマッチングでクリボーの画面座標を取得する。
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, position = cv2.minMaxLoc(result)

        if score >= MATCH_THRESHOLD:
            goomba_x = position[0]
        else:
            # 前フレームの検出位置を使い続けない。
            goomba_x = None

        if terminated or truncated:
            input("Press Enter to continue...")
            observation, info = env.reset()
            goomba_x = None
            former_x_pos = None
            stuck_frame_count = 0
except KeyboardInterrupt:
    pass
finally:
    env.close()
