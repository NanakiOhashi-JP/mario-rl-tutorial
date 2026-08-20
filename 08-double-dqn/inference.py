import time

import torch

from main import MODEL_PATH, QNetwork, get_device, make_env, states_to_tensor


def load_q_network(env, device, model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"学習済みモデルが見つかりません: {model_path}\n"
            "先に main.py を実行してモデルを作成してください。"
        )

    q_network = QNetwork(n_actions=env.action_space.n).to(device)
    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    q_network.load_state_dict(state_dict)
    q_network.eval()
    return q_network


def play(env, q_network, device, max_steps=10_000):
    state, info = env.reset()

    try:
        for _ in range(max_steps):
            state_tensor = states_to_tensor([state], device)

            with torch.no_grad():
                q_values = q_network(state_tensor)

            action = q_values.argmax(dim=1).item()
            next_state, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(4 / 60)

            if terminated or truncated:
                print(
                    f"time={info.get('time', 0)} "
                    f"score={info.get('score', 0)} "
                    f"x={info.get('x_pos', 0)}"
                )
                state, info = env.reset()
            else:
                state = next_state
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def main():
    device = get_device()
    env = make_env(render_mode="human")
    q_network = load_q_network(env, device)

    print(f"使用デバイス: {device}")
    print(f"読み込んだモデル: {MODEL_PATH}")

    play(env, q_network, device)


if __name__ == "__main__":
    main()
