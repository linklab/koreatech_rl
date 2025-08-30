from quanser.hardware import HIL
import numpy as np
card = HIL("qube_servo3_usb", "0")
led_channels = np.array([11000, 11001, 11002], dtype=np.uint32)

from _06_dqn_rip.a_quanser_env import QuanserEnv
from _07_ddpg_rip.a_ddpg_train_test import DDPGTester

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

    env = QuanserEnv(card, env_config)

    print(env.observation_space)
    print(env.action_space)
    model_path = "/home/hyoseok/src/koreatech_rl/_07_ddpg_rip/models/ddpg_quanser_3912.5_actor_20250830_212325.pth"
    ddpg = DDPGTester(env=env, num_episodes=30, model_path=model_path)
    ddpg.test()

if __name__ == "__main__":
    main()