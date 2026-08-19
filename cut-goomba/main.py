from pathlib import Path

import cv2
import gymnasium as gym
import gym_super_mario_bros  # noqa: F401: Mario環境をGymnasiumに登録する
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


WINDOW_NAME = "Goomba template cutter"
PREVIEW_NAME = "Template preview - Y: save / N: retry"
SCALE = 4
TEMPLATE_SIZE = 16
FRAME_DELAY_MS = 30
AUTO_PAUSE_AFTER_FRAMES = 20

OUTPUT_DIR = Path(__file__).resolve().parent / "templates"


def next_output_path() -> Path:
    """Return a new path without overwriting an existing template."""
    index = 1
    while True:
        path = OUTPUT_DIR / f"goomba_{index}.png"
        if not path.exists():
            return path
        index += 1


def crop_template(frame, selection):
    """Create a 16x16 crop centered on a selection from the scaled preview."""
    x, y, width, height = selection
    if width == 0 or height == 0:
        return None

    center_x = round((x + width / 2) / SCALE)
    center_y = round((y + height / 2) / SCALE)

    left = center_x - TEMPLATE_SIZE // 2
    top = center_y - TEMPLATE_SIZE // 2
    right = left + TEMPLATE_SIZE
    bottom = top + TEMPLATE_SIZE

    frame_height, frame_width = frame.shape[:2]
    if left < 0 or top < 0 or right > frame_width or bottom > frame_height:
        return None

    return frame[top:bottom, left:right].copy()


def select_and_save(frame) -> None:
    selection = cv2.selectROI(
        WINDOW_NAME,
        cv2.resize(
            frame,
            None,
            fx=SCALE,
            fy=SCALE,
            interpolation=cv2.INTER_NEAREST,
        ),
        showCrosshair=True,
        fromCenter=False,
    )

    template = crop_template(frame, selection)
    if template is None:
        print("Selection cancelled or too close to the image edge.")
        return

    preview = cv2.resize(
        template,
        None,
        fx=12,
        fy=12,
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imshow(PREVIEW_NAME, preview)
    print("Press Y to save this template, or N to retry.")

    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(PREVIEW_NAME)

    if key != ord("y"):
        print("Template was not saved.")
        return

    output_path = next_output_path()
    cv2.imwrite(str(output_path), template)
    print(f"Saved: {output_path} ({TEMPLATE_SIZE}x{TEMPLATE_SIZE})")


def add_instructions(frame, paused: bool):
    preview = cv2.resize(
        frame,
        None,
        fx=SCALE,
        fy=SCALE,
        interpolation=cv2.INTER_NEAREST,
    )

    status = "PAUSED" if paused else "RUNNING"
    lines = (
        f"{status}  S: select  SPACE: pause  N: next frame  Q: quit",
        "Drag around the Goomba, then press ENTER.",
    )

    for index, line in enumerate(lines):
        position = (12, 28 + index * 26)
        cv2.putText(
            preview,
            line,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            line,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return preview


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    env = gym.make(
        "SuperMarioBros-1-1-v0",
        render_mode="rgb_array",
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    observation, info = env.reset()
    paused = False
    goomba_visible_frames = 0
    has_auto_paused = False

    print("The game pauses automatically when the first Goomba is visible.")
    print("Controls: S select, SPACE pause, N next frame, Q quit")

    try:
        while True:
            frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
            cv2.imshow(WINDOW_NAME, add_instructions(frame, paused))

            key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF

            if key == ord("q"):
                break
            if key == ord("s"):
                paused = True
                select_and_save(frame)
                continue
            if key == ord(" "):
                paused = not paused

            advance_one_frame = key == ord("n")
            if paused and not advance_one_frame:
                continue

            observation, _, terminated, truncated, info = env.step(1)

            if 6 in info["enemy_types"]:
                goomba_visible_frames += 1
            else:
                goomba_visible_frames = 0

            if (
                not has_auto_paused
                and goomba_visible_frames >= AUTO_PAUSE_AFTER_FRAMES
            ):
                paused = True
                has_auto_paused = True
                print("Goomba detected. Paused for template selection.")

            if terminated or truncated:
                observation, info = env.reset()
                paused = False
                goomba_visible_frames = 0
                has_auto_paused = False
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
