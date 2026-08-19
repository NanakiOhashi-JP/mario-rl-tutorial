import time

import torch

from main import QNetwork, make_env, to_tensor


def play(env, q_network, max_steps=1000):
    observation, info = env.reset()
    q_network.eval()

    try:
        for _ in range(max_steps):
            state = to_tensor(observation)

            with torch.no_grad():
                q_values = q_network(state)

            action = q_values.argmax(dim=1).item()
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(1 / 60)

            if terminated or truncated:
                observation, info = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def main():
    env = make_env(render_mode="human")
    q_network = QNetwork(n_actions=env.action_space.n)
    play(env, q_network)


if __name__ == "__main__":
    main()
