import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
from env.env_utils import make_highway_weather
import numpy as np

def main():
    env = make_highway_weather(
        "composite-intersection-segment-v0",
        weather_probs=[0.7, 0.1, 0.1, 0.1],
        env_config={
            "observation": {
                "type": "Kinematics",
                "order": "sorted",
                "vehicles_count": 3,
                "see_behind": True,
                "features": ["presence", "x", "y", "vx", "vy", "long_off", "lat_off"],
            },
            "action": {
                "type": "ContinuousAction",
                "dynamical": True,
            },
            "duration": 200,
            "simulation_frequency": 15,
            "policy_frequency": 5,
        },
        seed=0,
    )

    obs, info = env.reset()
    done = False
    frame = env.render()
    height, width = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (width, height))

    step = 0
    truncated = False
    while not (done or truncated):
        action = np.array([0.1, 0.0], dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
        frame = env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
        print(f"step={step} reward={reward:.4f} done={done} truncated={truncated}")
        step += 1

    video_writer.release()
    env.close()


if __name__ == "__main__":
    main()
