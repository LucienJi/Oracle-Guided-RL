from env.env_utils import make_hybrid_walker
from env.env_utils import make_hybrid_lander
from env.env_utils import make_hybrid_car_racing
from env.env_utils import make_hybrid_weather
from third_party.customized_box2d.HybridBipedalWalker import HybridBipedalWalker
from dm_control import suite
import imageio



walker_env = make_hybrid_walker(mode="jumper",stack_frames=4,max_steps=1000)
frames = []
obs, info = walker_env.reset()
print(obs.shape)
for _ in range(100):
    action = walker_env.action_space.sample()
    obs, reward, done, truncated, info = walker_env.step(action)
    print("OBS: ", obs.shape)
    frame = walker_env.render()  # Should return an RGB array when render_mode="rgb_array"
    frames.append(frame)
    if done or truncated:
        obs, info = walker_env.reset()
walker_env.close()

# Save frames as video
video_path = "hybrid_walker_episode.mp4"
imageio.mimsave(video_path, frames, fps=30)
print(f"Video saved to {video_path}")


