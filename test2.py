from env.env_utils import make_hybrid_walker
from env.env_utils import make_hybrid_lander
from env.env_utils import make_hybrid_car_racing
from env.env_utils import make_hybrid_weather
from third_party.customized_box2d.HybridBipedalWalker import HybridBipedalWalker
from dm_control import suite
import imageio
import math
import numpy as np


# walker_env = make_hybrid_walker(mode="jumper",stack_frames=1,max_steps=1000)
from third_party.customized_box2d.HybridBipedalWalker import N_LIDAR_RAYS, LIDAR_ANGLE_START, LIDAR_ANGLE_END
def run_scripted_sanity(seed=0, render=True, mode="jumper", max_steps=2000, save_video=False, video_path=None):
    # 你已经定义了 HybridBipedalWalker 类的话，直接用
    # Use rgb_array mode if saving video, otherwise use human or None
    render_mode = "rgb_array"
    
    env = HybridBipedalWalker(render_mode=render_mode, mode=mode)
    
    # Collect frames for video if saving
    frames = [] if save_video else None

    obs, _ = env.reset(seed=seed)

    # 计算哪些 lidar rays 是“朝前”的（用 sin(angle)>0 来判断 x 方向向前）
    angles = []
    for i in range(N_LIDAR_RAYS):
        t = i / (N_LIDAR_RAYS - 1)
        angle = LIDAR_ANGLE_START + t * (LIDAR_ANGLE_END - LIDAR_ANGLE_START)
        angles.append(angle)
    forward_idx = [i for i, a in enumerate(angles) if math.sin(a) > 0.2]  # >0.2 更“向前”
    if len(forward_idx) == 0:
        forward_idx = list(range(N_LIDAR_RAYS // 2, N_LIDAR_RAYS))

    def to_act(cur_angle, tgt_angle, k=4.0):
        # action 直接乘 motorSpeed；用角度误差做一个很粗的 P 控制
        return float(np.clip(k * (tgt_angle - cur_angle), -1.0, 1.0))

    phase = 0.0
    jump_state = "run"
    timer = 0

    print("forward_idx:", forward_idx[:5], "...", forward_idx[-5:])

    for t in range(max_steps):
        # 解析关节角（和你 env.step() 里 state 对应）
        hipL = float(obs[4])
        kneeL = float(obs[6] - 1.0)   # 你存的是 joint.angle + 1.0
        hipR = float(obs[10])
        kneeR = float(obs[12] - 1.0)

        lidar = np.array(obs[14:14 + N_LIDAR_RAYS], dtype=np.float32)

        fwd = lidar[forward_idx]
        fwd_mean = float(np.mean(fwd))
        fwd_min = float(np.min(fwd))

        # “坑在前方” 的一个简单判定：前向 rays 大部分都接近 1（ray 没打到地）
        pit_ahead = (fwd_mean > 0.98) and (fwd_min > 0.95)

        # 状态机：run -> crouch -> extend -> fly -> run
        if jump_state == "run" and pit_ahead:
            jump_state = "crouch"
            timer = 12  # 蹲 12 步

        if jump_state == "crouch":
            # 蹲：膝盖更弯（更负），身体稍微前倾
            tgt_hip_L = tgt_hip_R = 0.2
            tgt_knee_L = tgt_knee_R = -1.30
            timer -= 1
            if timer <= 0:
                jump_state = "extend"
                timer = 6  # 蹬 6 步

        elif jump_state == "extend":
            # 蹬：膝盖更直（接近上限 -0.1），身体前倾一点给水平速度
            tgt_hip_L = tgt_hip_R = 0.6
            tgt_knee_L = tgt_knee_R = -0.15
            timer -= 1
            if timer <= 0:
                jump_state = "fly"
                timer = 12  # 空中/落地缓冲

        elif jump_state == "fly":
            # 空中/落地：尽量稳一点
            tgt_hip_L = tgt_hip_R = 0.0
            tgt_knee_L = tgt_knee_R = -0.20
            timer -= 1
            if timer <= 0:
                jump_state = "run"

        else:
            # run：非常粗的交替步态（不保证优雅，但能跑起来）
            phase += 0.15
            hip_amp = 0.7
            knee_base = -0.9
            knee_amp = 0.6

            tgt_hip_L = hip_amp * math.sin(phase)
            tgt_hip_R = hip_amp * math.sin(phase + math.pi)

            tgt_knee_L = knee_base + knee_amp * math.sin(phase + math.pi / 2)
            tgt_knee_R = knee_base + knee_amp * math.sin(phase + math.pi / 2 + math.pi)

        action = np.array([
            to_act(hipL, tgt_hip_L, k=2.0),
            to_act(kneeL, tgt_knee_L, k=6.0),
            to_act(hipR, tgt_hip_R, k=2.0),
            to_act(kneeR, tgt_knee_R, k=6.0),
        ], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Capture frame for video if saving
        if save_video:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if t % 25 == 0:
            pos = env.hull.position
            vel = env.hull.linearVelocity
            print(f"t={t:4d} x={pos.x:6.2f} y={pos.y:5.2f} vx={vel.x:5.2f} "
                  f"pit={pit_ahead} fwd_mean={fwd_mean:.3f} fwd_min={fwd_min:.3f} state={jump_state}")

        if terminated or truncated:
            pos = env.hull.position
            print("DONE:", "terminated" if terminated else "truncated", "info=", info,
                  f"final_pos=({pos.x:.2f},{pos.y:.2f})")
            break

    env.close()
    
    # Save video if frames were collected
    if save_video and frames:
        if video_path is None:
            video_path = f"hybrid_walker_{mode}_seed{seed}.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video saved to {video_path}")


if __name__ == "__main__":
    # 建议先用 jumper（全是 pit）
    run_scripted_sanity(seed=0, render=False, mode="jumper", max_steps=2000, save_video=True, video_path="hybrid_walker_episode.mp4")
