from quanser.hardware import HIL
import numpy as np
card = HIL("qube_servo3_usb", "0")
led_channels = np.array([11000, 11001, 11002], dtype=np.uint32)

from _06_dqn_rip.a_quanser_env import QuanserEnv
from _07_ddpg_rip.a_ddpg_train_test import DDPGTrainer

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
    config = {
        "env_name": "quanser",                            # 환경의 이름
        "max_num_episodes": 200_000,                      # 최대 학습 에피소드 수
        "batch_size": 128,                                # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "replay_buffer_size": 100_000,                    # 리플레이 버퍼 사이즈
        "learning_rate": 0.001,                           # 학습율
        "gamma": 0.999,                                   # 감가율
        "soft_update_tau": 0.005,                         # 타깃 네트워크 소프트 업데이트 계수 τ
        "print_episode_interval": 1,                      # Episode 통계 출력에 관한 에피소드 간격
        "episode_reward_avg_solved": 3800.0,              # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
        "training_start_steps": 128,                      # 학습 시작 최소 스텝(버퍼 워밍업)
        "forward_start_episode": 50,                      # 정책 실행 시작 에피소드
        "train_per_step": 2,                              # 환경 스텝당 학습 수
        "step_control_time": 0.006,                       # 한 스텝 제어 간격(초)
        "clip_grad_max_norm": 1.0,                        # 그래디언트 클리핑 최대 노름
        "validation_num_episodes": 3,                     # 검증 에피소드 수
        "valiation_interval": 10,                         # 검증 간격(에피소드 수)
        "validation_start_episode": 100,                  # 검증 시작 에피소드
        "validation_start_gap_reward": 300.0,             # 검증 조기시작 기준(목표-현재 갭)
        "start_noise_scale": 1.0,                         # 정책 시작 노이즈 스케일
        "end_noise_scale": 0.1,                           # 정책 종료 노이즈 스케일
        "noise_decay_steps": 100_000,                     # 노이즈 스케일 선형 감소 스텝 수
        "noise_keep_steps": 5,                            # 노이즈 스케일 유지 스텝 수
        "policy_update_interval": 2,                      # 정책 업데이트 간격(스텝)
        "target_policy_noise": 0.2,                       # 타깃 정책 스무딩 노이즈
        "target_noise_clip": 0.5                          # 타깃 정책 노이즈 클리핑 값
    }
    env = QuanserEnv(card, env_config)

    print(env.observation_space)
    print(env.action_space)
    use_wandb = True
    ddpg = DDPGTrainer(env=env, config=config, use_wandb=use_wandb)
    ddpg.train_loop()

if __name__ == "__main__":
    main()