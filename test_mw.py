import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import imageio
import numpy as np
import torch

from third_party.custimized_metaworld.oracle_factory_th import get_oracle_pair
from env.env_utils import make_metaworld_env

ENV_NAMES = [
    "assembly", # 0.2 ; 0.1
    "basketball", # 0.2 ; 0.1
    "drawer-open", # 0.3 ; 0.3
    "hammer", # 0.3 ; 0.3
    "lever-pull", # 0.1 ; 0.1
    "peg-insert-side", # 0.1 ; 0.1
    "peg-unplug-side", # 0.1 ; 0.1
    "pick-place", # 0.2 ; 0.3
    "push-wall", # 0.2 ; 0.2
    "stick-pull", # 0.1 ; 0.1
]
MODES = ["condition"]
MIN_NOISE = 0.03
SEEDS = range(5)
SAVE_VIDEO = True  # Set to False to skip rendering and video saving

# Parse noise scales from comments: format is "# max_noise0 ; max_noise1"
ENV_NOISE_SCALES = {
    "assembly": (0.1, 0.1),
    "basketball": (0.2, 0.1),
    "drawer-open": (0.5, 0.5),
    "hammer": (0.3, 0.3),
    "lever-pull": (0.08, 0.08),
    "peg-insert-side": (0.06, 0.06),
    "peg-unplug-side": (0.1, 0.1),
    "pick-place": (0.1, 0.2),
    "push-wall": (0.1, 0.1),
    "stick-pull": (0.1, 0.1),
}


def rollout_episode(env, oracle, seed: int, save_video: bool = True):
    obs, info = env.reset(seed=seed)
    frames = []
    done = False
    truncated = False
    max_steps = getattr(getattr(env, "unwrapped", env), "max_path_length", 500)
    steps = 0
    total_reward = 0.0
    
    while not (done or truncated) and steps < max_steps:
        # Convert numpy obs to torch tensor
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        # Get action from torch policy
        action_t = oracle.get_action(obs_t)
        # Convert back to numpy and ensure 1D
        action = action_t.detach().cpu().numpy()
        action = action.squeeze()  # Remove batch dimension if present, ensure 1D
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if save_video:
            frame = env.render()
            if frame is not None:
                frame = np.flipud(frame)
                frames.append(frame)
        if info.get("success", False) or info.get("is_success", False):
            done = True
        steps += 1
    
    success = bool(info.get("success", False) or info.get("is_success", False))
    episode_return = total_reward
    episode_length = steps
    
    # Get return from RecordEpisodeStatistics if available
    if "episode" in info:
        episode_return = info["episode"]["r"]
        episode_length = info["episode"]["l"]
    
    return success, episode_return, episode_length, frames


def main():
    if SAVE_VIDEO:
        os.makedirs("debug_videos", exist_ok=True)
    
    # Create a temporary env to get obs_dim for obs_config
    temp_env = gym.make("Meta-World/MT1", env_name="assembly-v3", seed=0)
    obs_dim = temp_env.observation_space.shape[0]
    temp_env.close()
    obs_config = {"robot_states": [obs_dim]}
    
    for env_name in ENV_NAMES:
        env_id = f"{env_name}-v3"
        max_noise0, max_noise1 = ENV_NOISE_SCALES[env_name]
        noise_scales0 = (MIN_NOISE, max_noise0)
        noise_scales1 = (MIN_NOISE, max_noise1)
        
        for mode in MODES:
            # Test oracle0
            oracle0 = get_oracle_pair(
                env_name,
                mode=mode,
                noise_scales=noise_scales0,
                seed=0,
            )[0]
            
            success_count = 0
            total_return = 0.0
            total_length = 0
            all_frames = []
            for seed in SEEDS:
                env = make_metaworld_env(
                    env_name=env_id,
                    obs_config=obs_config,
                    seed=seed,
                    camera_to_render="corner4",
                    reward_scale=1.0,   
                    step_penality=0.05,
                    max_steps=400,
                )
                success, episode_return, episode_length, frames = rollout_episode(env, oracle0, seed, save_video=SAVE_VIDEO)
                env.close()
                success_count += int(success)
                total_return += episode_return
                total_length += episode_length
                if SAVE_VIDEO:
                    all_frames.extend(frames)
                print(
                    f"[{env_name}][{mode}][ns={noise_scales0}] "
                    f"oracle=0 seed={seed} success={success} "
                    f"return={episode_return:.2f} length={episode_length}"
                )

            success_rate = success_count / len(SEEDS)
            avg_return = total_return / len(SEEDS)
            avg_length = total_length / len(SEEDS)
            if SAVE_VIDEO:
                video_path = os.path.join(
                    "debug_videos",
                    f"{env_id}_{mode}_oracle0_seeds{len(SEEDS)}"
                    f"_ns{noise_scales0[0]}_{noise_scales0[1]}_sr{success_rate:.2f}.mp4",
                )
                imageio.mimsave(video_path, all_frames, fps=30)
                print(
                    f"[{env_name}][{mode}][ns={noise_scales0}] "
                    f"oracle=0 success_rate={success_rate:.2f} "
                    f"avg_return={avg_return:.2f} avg_length={avg_length:.1f} "
                    f"video={video_path}"
                )
            else:
                print(
                    f"[{env_name}][{mode}][ns={noise_scales0}] "
                    f"oracle=0 success_rate={success_rate:.2f} "
                    f"avg_return={avg_return:.2f} avg_length={avg_length:.1f}"
                )
            
            # Test oracle1
            oracle1 = get_oracle_pair(
                env_name,
                mode=mode,
                noise_scales=noise_scales1,
                seed=0,
            )[1]
            
            success_count = 0
            total_return = 0.0
            total_length = 0
            all_frames = []
            for seed in SEEDS:
                env = make_metaworld_env(
                    env_name=env_id,
                    obs_config=obs_config,
                    seed=seed,
                    camera_to_render="corner4",
                    reward_scale=1.0,
                    step_penality=0.05,
                    max_steps=400,
                )
                success, episode_return, episode_length, frames = rollout_episode(env, oracle1, seed, save_video=SAVE_VIDEO)
                env.close()
                success_count += int(success)
                total_return += episode_return
                total_length += episode_length
                if SAVE_VIDEO:
                    all_frames.extend(frames)
                print(
                    f"[{env_name}][{mode}][ns={noise_scales1}] "
                    f"oracle=1 seed={seed} success={success} "
                    f"return={episode_return:.2f} length={episode_length}"
                )

            success_rate = success_count / len(SEEDS)
            avg_return = total_return / len(SEEDS)
            avg_length = total_length / len(SEEDS)
            if SAVE_VIDEO:
                video_path = os.path.join(
                    "debug_videos",
                    f"{env_id}_{mode}_oracle1_seeds{len(SEEDS)}"
                    f"_ns{noise_scales1[0]}_{noise_scales1[1]}_sr{success_rate:.2f}.mp4",
                )
                imageio.mimsave(video_path, all_frames, fps=30)
                print(
                    f"[{env_name}][{mode}][ns={noise_scales1}] "
                    f"oracle=1 success_rate={success_rate:.2f} "
                    f"avg_return={avg_return:.2f} avg_length={avg_length:.1f} "
                    f"video={video_path}"
                )
            else:
                print(
                    f"[{env_name}][{mode}][ns={noise_scales1}] "
                    f"oracle=1 success_rate={success_rate:.2f} "
                    f"avg_return={avg_return:.2f} avg_length={avg_length:.1f}"
                )


if __name__ == "__main__":
    main()