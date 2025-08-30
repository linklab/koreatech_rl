from quanser.hardware import HIL
import numpy as np
card = HIL("qube_servo3_usb", "0")
led_channels = np.array([11000, 11001, 11002], dtype=np.uint32)

from _06_dqn_rip.a_quanser_env import QuanserEnv
from _07_ddpg_rip.a_ddpg_train_test import DDPGTester

def main():
    env_config = {
        "episode_max_steps": 2000,               # 에피소드당 최대 스텝 수
        "pwm_action_scale": 0.35,                # PWM 액션 스케일
        "Kp": 0.8,                               # P 게인 (비례 제어 이득)
        "Kd": 0.02,                              # D 게인 (미분 제어 이득)
        "motor_angle_theory_max": 2.5,           # 모터 각도 이론값 최대
        "motor_angle_theory_min": -2.5,          # 모터 각도 이론값 최소
        "motor_angle_obs_max": 1.8,              # 모터 각도 관찰값 최대
        "sinusoidal_pendulum_angle_max": 1.0,    # 정현파 펜듈럼 각도 최대
        "sinusoidal_pendulum_angle_min": -1.0,   # 정현파 펜듈럼 각도 최소
        "motor_angle_vel_max": 20.0,             # 모터 각속도 최대
        "pendulum_angle_vel_max": 40.0,          # 펜듈럼 각속도 최대
        "pendulum_spin_num_max": 5.0,            # 펜듈럼 회전 횟수 최대
        "num_actions": 5,                        # 이산 액션 수
        "action_min": -1.0,                      # 연속 액션 최소
        "action_max": 1.0,                       # 연속 액션 최대
        "motor_encoder_max_count": 2048.0,       # 모터 엔코더 최대 카운트
        "reset_motor_init_count_force": 0.05,    # 모터 초기 카운트 재설정 힘
        "reset_motor_init_count_time": 3.0,      # 모터 초기 카운트 재설정 힘주는 시간
        "reset_motor_init_count_dt": 0.01,       # 모터 초기 카운트 재설정 간격
        "reset_motor_init_count_interval": 5,    # 모터 초기 카운트 재설정 에피소드 간격
        "reset_pendulum_init_count_mean": 100,   # 펜듈럼 초기 카운트 평균내는 수
        "reset_pendulum_init_count_dt": 0.003,   # 펜듈럼 초기 카운트 간격
        "reset_success_num": 50,                 # 연속 리셋 성공 횟수(만족시 리셋 성공 판정)
        "reset_pwm": 0.04,                       # 리셋 제어용 PWM 값
        "reset_dt": 0.005,                       # 리셋 제어 간격
        "reset_threshold_pendulum_vel": 0.1,     # 리셋 성공 판정 펜듈럼 각속도 임계값
        "reset_threshold_motor_rad": 0.1,        # 리셋 성공 판정 모터 각도 임계값
        "reset_fail_count": 1000,                # 연속 리셋 실패 횟수(만족시 리셋 실패 판정 후 임계값 완화)
        "reset_fail_rad": 0.3,                   # 리셋 실패 시 모터 각도 임계값(만족시 모터 초기 카운트 재설정)
        "reset_helper_success_num": 150,         # 연속 리셋 헬퍼 성공 횟수
        "reset_helper_threshold_motor_rad": 0.2, # 리셋 헬퍼 성공 판정 모터 각도 임계값
        "reset_helper_max_count": 3000,          # 리셋 헬퍼 최대 시도 횟수
        "reset_helper_fast_angle": 90,           # 리셋 헬퍼 빠른 PWM 제어 조건 각도
        "reset_helper_fast_pwm": 0.4,            # 리셋 헬퍼 빠른 PWM 제어값
        "reset_helper_pwm": 0.2,                 # 리셋 헬퍼 PWM 제어값
        "reset_helper_dt": 0.005,                # 리셋 헬퍼 제어 간격
        "double_reward_deg": 170,                # 보상 두배 조건 각도
        "terminate_pendulum_spin_num": 5.0,      # 종료 조건 펜듈럼 회전 횟수
        "terminate_motor_angle_deg": 90.0,       # 종료 조건 모터 각도
        "terminate_pendulum_angle_vel": 40.0     # 종료 조건 펜듈럼 각속도
    }

    env = QuanserEnv(card, env_config)

    print(env.observation_space)
    print(env.action_space)
    model_path = "/home/hyoseok/src/koreatech_rl/_07_ddpg_rip/models/ddpg_quanser_3912.5_actor_20250830_212325.pth"
    ddpg = DDPGTester(env=env, num_episodes=30, model_path=model_path)
    ddpg.test()

if __name__ == "__main__":
    main()