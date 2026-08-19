import argparse
import time
from collections import Counter

import torch

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from inference import load_q_network
from main import MODEL_PATH, get_device, make_env, states_to_tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="ウィンドウを表示せず、学習済みのマリオを動かします。"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10_000,
        help="推論する最大ステップ数（デフォルト: 10000）",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        help="途中経過を表示する間隔（デフォルト: 25）",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="1ステップごとに待つ秒数（デフォルト: 0）",
    )
    return parser.parse_args()


def format_action(action):
    buttons = "+".join(SIMPLE_MOVEMENT[action])
    return f"{action}:{buttons}"


def print_episode_result(episode, reward, max_x_pos, info, action_counts):
    result = "クリア" if info.get("flag_get", False) else "終了"
    actions = ", ".join(
        f"{format_action(action)}={count}回"
        for action, count in action_counts.most_common()
    )

    print(
        f"episode={episode} result={result} "
        f"reward={reward:.1f} max_x={max_x_pos} "
        f"time={info.get('time', '--')} score={info.get('score', '--')}"
    )
    print(f"actions: {actions}")


def play(env, q_network, device, max_steps, log_interval, delay):
    if max_steps <= 0:
        raise ValueError("--max-stepsには1以上を指定してください")
    if log_interval <= 0:
        raise ValueError("--log-intervalには1以上を指定してください")
    if delay < 0:
        raise ValueError("--delayには0以上を指定してください")

    state, info = env.reset()
    episode = 1
    episode_reward = 0.0
    max_x_pos = info.get("x_pos", 0)
    action_counts = Counter()

    try:
        for step in range(1, max_steps + 1):
            state_tensor = states_to_tensor([state], device)

            with torch.no_grad():
                q_values = q_network(state_tensor)

            action = q_values.argmax(dim=1).item()
            next_state, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            max_x_pos = max(max_x_pos, info.get("x_pos", 0))
            action_counts[action] += 1

            if step % log_interval == 0:
                print(
                    f"step={step} episode={episode} "
                    f"x={info.get('x_pos', '--')} "
                    f"time={info.get('time', '--')} "
                    f"score={info.get('score', '--')} "
                    f"action={format_action(action)} "
                    f"q={q_values[0, action].item():.2f}"
                )

            if delay > 0:
                time.sleep(delay)

            if terminated or truncated:
                print_episode_result(
                    episode,
                    episode_reward,
                    max_x_pos,
                    info,
                    action_counts,
                )
                state, info = env.reset()
                episode += 1
                episode_reward = 0.0
                max_x_pos = info.get("x_pos", 0)
                action_counts.clear()
            else:
                state = next_state
    except KeyboardInterrupt:
        print("\n推論を中断しました")
    finally:
        env.close()


def main():
    args = parse_args()
    device = get_device()
    env = make_env(render_mode="rgb_array")
    q_network = load_q_network(env, device)

    print(f"使用デバイス: {device}")
    print(f"読み込んだモデル: {MODEL_PATH}")
    print("画面は表示せず、ターミナルへ途中経過を表示します")

    play(
        env,
        q_network,
        device,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
