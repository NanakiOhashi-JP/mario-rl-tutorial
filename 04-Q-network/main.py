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
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=False)
    env = FrameStackObservation(env, stack_size=4)
    return env


def to_tensor(observation):
    observation = torch.as_tensor(observation, dtype=torch.float32) / 255.0
    return observation.unsqueeze(0)


def main():
    env = make_env()
    observation, info = env.reset()

    state = to_tensor(observation)
    q_network = QNetwork(n_actions=env.action_space.n)

    with torch.no_grad():
        q_values = q_network(state)

    action = q_values.argmax(dim=1).item()

    print(f"入力の形: {state.shape}")
    print(f"出力の形: {q_values.shape}")
    print(f"Q値: {q_values}")
    print(f"選んだ行動: {action} {SIMPLE_MOVEMENT[action]}")

    env.close()


if __name__ == "__main__":
    main()
