import gymnasium as gym
import cv2
import highway_env
from env.env_utils import make_highway_weather
from env.env_utils import make_dmc_env
from dm_control import suite

import numpy as np

# env = suite.load(domain_name="hopper", task_name="stand")
# obs = env.reset()
# print(obs)
# done = False
# breakpoint()

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_writer = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (256, 256))

# while not done:
#     action = env.action_spec().generate_value()
#     obs, reward, done, truncated, info = env.step(action)
#     frame = env.render()
#     # Convert RGB to BGR for OpenCV
#     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     video_writer.write(frame_bgr)
#     print(ct,done,truncated)
#     ct += 1
# video_writer.release()
# env.close()
env = make_highway_weather("composite-intersection-segment-v0", weather_probs=[0.7, 0.1, 0.1, 0.1],
                           env_config = {
                               'observation':{
                                   'type': 'Kinematics',
                                   'order':'sorted',
                                   'vehicles_count': 3,
                                   'see_behind': True,
                                   'features': ['presence', 'x', 'y', 'vx', 'vy','long_off','lat_off'],
                               },
                               'action':{
                                   'type': 'ContinuousAction',
                                   'dynamical': True,
                               },
                               "duration": 200,
                                "simulation_frequency": 15,  # [Hz]
                                "policy_frequency": 5,  # [Hz]
                           },
                           seed=0)

obs, info = env.reset()
done = False
# Get the first frame to determine video dimensions
frame = env.render()
height, width = frame.shape[:2]

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (width, height))

ct = 0
truncated = False
while not (done or truncated):
    action = np.array([0.1, 0.0])
    obs, reward, done, truncated, info = env.step(action)
    breakpoint()
    frame = env.render()
    # Convert RGB to BGR for OpenCV
    print("REWARD: ", reward)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(frame_bgr)
    print(ct,done,truncated)
    ct += 1

video_writer.release()
env.close()