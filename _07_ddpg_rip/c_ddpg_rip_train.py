from quanser.hardware import HIL
import numpy as np
card = HIL("qube_servo3_usb", "0")
led_channels = np.array([11000, 11001, 11002], dtype=np.uint32)

from _06_dqn_rip.a_quanser_env import QuanserEnv
from _07_ddpg_rip.a_ddpg_train_test import DDPGTrainer

def main():
    env_config = {
        "episode_max_steps": 2000,        # 에피소드당 최대 스텝 수
        "pwm_action_scale": 0.35, #
        "Kp": 0.8, #
        "Kd": 0.02, #
        "motor_angle_theory_max": 2.5, #
        "motor_angle_theory_min": -2.5, #
        "motor_angle_obs_max": 1.8, #
        "sinusoidal_pendulum_angle_max": 1.0, #
        "sinusoidal_pendulum_angle_min": -1.0, #
        "motor_angle_vel_max": 20.0, #
        "pendulum_angle_vel_max": 40.0, #
        "pendulum_spin_num_max": 5.0, #
        "num_actions": 5, #
        "action_min": -1.0, #
        "action_max": 1.0, #
        "motor_encoder_max_count": 2048.0, #
        "reset_motor_init_count_force": 0.05, #
        "reset_motor_init_count_time": 3.0, #
        "reset_motor_init_count_dt": 0.01, #
        "reset_motor_init_count_interval": 5, #
        "reset_pendulum_init_count_mean": 100, #
        "reset_pendulum_init_count_dt": 0.003, #
        "reset_success_num": 50, #
        "reset_pwm": 0.04, #
        "reset_dt": 0.005, #
        "reset_threshold_pendulum_vel": 0.1, #
        "reset_threshold_motor_rad": 0.1, #
        "reset_fail_count": 1000, #
        "reset_fail_rad": 0.3, #
        "reset_helper_success_num": 150, #
        "reset_helper_threshold_motor_rad": 0.2, #
        "reset_helper_max_count": 3000, #
        "reset_helper_fast_angle": 90, #
        "reset_helper_fast_pwm": 0.4,
        "reset_helper_pwm": 0.2,
        "reset_helper_dt": 0.005,
        "double_reward_deg": 170,
        "terminate_pendulum_spin_num": 5.0,
        "terminate_motor_angle_deg": 90.0,
        "terminate_pendulum_angle_vel": 40.0
    }
    config = {
        "env_name": "quanser",                              # 환경의 이름
        "max_num_episodes": 200_000,                        # 훈련을 위한 최대 에피소드 횟수
        "batch_size": 128,                                  # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "replay_buffer_size": 100_000,                    # 리플레이 버퍼 사이즈
        "learning_rate": 0.001,                            # 학습율
        "gamma": 0.999,                                      # 감가율
        "soft_update_tau": 0.005,                           # td3 Soft Update Tau
        "print_episode_interval": 1,                        # Episode 통계 출력에 관한 에피소드 간격
        "episode_reward_avg_solved": 3800.0,                   # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
        "training_start_steps": 128,
        "forward_start_episode": 50,
        "train_per_step": 2,
        "step_control_time": 0.006,
        "clip_grad_max_norm": 1.0,
        "validation_num_episodes": 3,
        "valiation_interval": 10,
        "validation_start_episode": 100,
        "validation_start_gap_reward": 300.0,
        "start_noise_scale": 1.0,
        "end_noise_scale": 0.1,
        "noise_decay_steps": 100_000,
        "noise_keep_steps": 5,
        "policy_update_interval": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5

    }
    env = QuanserEnv(card, env_config)

    print(env.observation_space)
    print(env.action_space)
    use_wandb = True
    ddpg = DDPGTrainer(env=env, config=config, use_wandb=use_wandb)
    ddpg.train_loop()

if __name__ == "__main__":
    main()