import gymnasium as gym
import gym_super_mario_bros

env = gym.make(
    "SuperMarioBros-1-1-v0",
    render_mode="human"
)

env.reset()

env.close()