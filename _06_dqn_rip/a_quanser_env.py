from quanser.hardware import MAX_STRING_LENGTH, HILError
from array   import array
import time
import math
import torch
import numpy as np
import gymnasium as gym
import copy

from collections import deque
np.set_printoptions(precision=5, suppress=True)


class QuanserEnv(gym.Env):
    def __init__(self, card, config):
        self.card = card

        # PWM mode setup
        self.card.set_card_specific_options("pwm_en=1", MAX_STRING_LENGTH)
        input_channels = array('I', [1])
        output_channels = array('I', [0])
        num_input_channels = len(input_channels)
        num_output_channels = len(output_channels)
        self.card.set_digital_directions(input_channels, num_input_channels, output_channels, num_output_channels)
        self.card.write_digital(array('I',[0]),1,array('I',[1]))

        # channels
        self.pwm_ch = array('I', [0]) # pwm output channel
        self.motor_enc_ch = array('I', [0]) # encoder input channel (motor angle)
        self.pend_enc_ch = array('I', [1]) # encoder input channel (pendulum angle)
        self.tach_ch = array('I', [14001]) # tachometer input channel (pendulum angular velocity)
        self.tach_motor_ch = array('I', [14000]) # tachometer input channel (motor angular velocity)

        # LED channels
        self.led_channels = np.array([11000, 11001, 11002], dtype=np.uint32)

        # encoder value buffer
        self.motor_enc_val = array('i', [0])  # motor angle
        self.pend_enc_val = array('i', [0])  # pendulum angle
        self.tach_vel_val = array('d', [0.0])  # pendulum angular velocity
        self.tach_motor_vel_val = array('d', [0.0])  # motor angular velocity

        # control parameters
        self.max_steps = config["episode_max_steps"]
        self.action_scale = config["pwm_action_scale"]

        # counter
        self.step_count = 0
        self.reset_count = 0

        # for PID control
        self.Kp = config["Kp"]
        self.Kd = config["Kd"]

        # for estimate angular velocity
        self.prev_motor_angle = None
        self.prev_pend_angle = None

        # for initialization pendulum count
        self.pen_init_count_list = []

        # configs
        self.motor_angle_theory_max = config["motor_angle_theory_max"]
        self.motor_angle_theory_min = config["motor_angle_theory_min"]
        self.motor_angle_obs_max = config["motor_angle_obs_max"]
        self.sinusoidal_pendulum_angle_max = config["sinusoidal_pendulum_angle_max"]
        self.sinusoidal_pendulum_angle_min = config["sinusoidal_pendulum_angle_min"]
        self.action_min = config["action_min"]
        self.action_max = config["action_max"]
        self.num_actions = config["num_actions"]
        self.motor_angle_vel_max = config["motor_angle_vel_max"]
        self.pendulum_angle_vel_max = config["pendulum_angle_vel_max"]
        self.pendulum_spin_num_max = config["pendulum_spin_num_max"]
        self.motor_encoder_max_count = config["motor_encoder_max_count"]
        self.reset_motor_init_count_force = config["reset_motor_init_count_force"]
        self.reset_motor_init_count_time = config["reset_motor_init_count_time"]
        self.reset_motor_init_count_dt = config["reset_motor_init_count_dt"]
        self.reset_motor_init_count_interval = config["reset_motor_init_count_interval"]
        self.reset_pendulum_init_count_mean = config["reset_pendulum_init_count_mean"]
        self.reset_pendulum_init_count_dt = config["reset_pendulum_init_count_dt"]
        self.reset_success_num = config["reset_success_num"]
        self.reset_pwm = config["reset_pwm"]
        self.reset_dt = config["reset_dt"]
        self.reset_threshold_pendulum_vel = config["reset_threshold_pendulum_vel"]
        self.reset_threshold_motor_rad = config["reset_threshold_motor_rad"]
        self.reset_fail_count = config["reset_fail_count"]
        self.reset_fail_rad = config["reset_fail_rad"]
        self.reset_helper_success_num = config["reset_helper_success_num"]
        self.reset_helper_threshold_motor_rad = config["reset_helper_threshold_motor_rad"]
        self.reset_helper_max_count = config["reset_helper_max_count"]
        self.reset_helper_fast_angle = config["reset_helper_fast_angle"]
        self.reset_helper_fast_pwm = config["reset_helper_fast_pwm"]
        self.reset_helper_pwm = config["reset_helper_pwm"]
        self.reset_helper_dt = config["reset_helper_dt"]
        self.double_reward_deg = config["double_reward_deg"]
        self.terminate_pendulum_spin_num = config["terminate_pendulum_spin_num"]
        self.terminate_motor_angle_deg = config["terminate_motor_angle_deg"]
        self.terminate_pendulum_angle_vel = config["terminate_pendulum_angle_vel"]

        low_obs = np.array([
            self.motor_angle_theory_min,          # motor_angle(radian)
            self.sinusoidal_pendulum_angle_min,   # sin(pendulum_angle(radian))
            self.sinusoidal_pendulum_angle_min,   # cos(pendulum_angle(radian))
            -np.inf,                              # motor_ang_vel(radian/sec)
            -np.inf,                              # pendulum_ang_vel(radian/sec)
            -np.inf                               # pendulum_spin_num(float)
        ], dtype=np.float32)

        high_obs = np.array([
            self.motor_angle_theory_max,
            self.sinusoidal_pendulum_angle_max,
            self.sinusoidal_pendulum_angle_max,
             np.inf,
             np.inf,
             np.inf
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=low_obs,
                                                high=high_obs,
                                                dtype=np.float32)
        
        # for DDPG
        self.action_space = gym.spaces.Box(low=np.array([self.action_min], dtype=np.float32),
                                           high=np.array([self.action_max], dtype=np.float32),
                                           dtype=np.float32)
        
        # for DQN
        self.dqn_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
        
        # reset encoder counts
        self.init_count = 0  # motor init count
        self.pend_init_count = 0  # pendulum init count

        # set initial motor count
        self._reset_motor_init_count()


    def observation_space(self):
        return self.observation_space

    def action_space(self):
        return self.action_space

    def _get_motor_angle(self):
        if self.prev_motor_angle is None:
            self.prev_motor_read = None
        else:
            self.prev_motor_read = copy.deepcopy(self.motor_read0)
        self.card.read_encoder(self.motor_enc_ch, 1, self.motor_enc_val)
        self.motor_read0 = time.perf_counter()
        count = self.motor_enc_val[0] - self.init_count
        return count * 2 * math.pi / self.motor_encoder_max_count # radian

    def _get_pendulum_angle(self):
        if self.prev_pend_angle is None:
            self.prev_pend_read = None
        else:
            self.prev_pend_read = copy.deepcopy(self.pendulum_read0)
        self.card.read_encoder(self.pend_enc_ch, 1, self.pend_enc_val)
        self.pendulum_read0 = time.perf_counter()
        raw_angle = (self.pend_enc_val[0] - self.pend_init_count) * 2 * math.pi / self.motor_encoder_max_count
        angle = ((raw_angle + math.pi) % (2 * math.pi)) - math.pi
        return angle  # radian [-π, π], 6 o'clock: 0 radian

    def _get_motor_velocity(self):
        self.card.read_other(self.tach_motor_ch, 1, self.tach_motor_vel_val)
        tach_motor_vel_rad = self.tach_motor_vel_val[0] * 2.0 * math.pi / self.motor_encoder_max_count
        return tach_motor_vel_rad

    def _get_pendulum_velocity(self):
        self.card.read_other(self.tach_ch, 1, self.tach_vel_val)
        tach_vel_rad = self.tach_vel_val[0] * 2.0 * math.pi / self.motor_encoder_max_count
        return tach_vel_rad
    
    def _get_motor_acceleration(self, cur_motor_vel):
        if self.prev_motor_vel is None:
            self.prev_motor_vel = cur_motor_vel
            return 0.0
        else:
            acc = (cur_motor_vel - self.prev_motor_vel)
            self.prev_motor_vel = cur_motor_vel
            return acc
        
    def _get_pend_acceleration(self, cur_pend_vel):
        if self.prev_pend_vel is None:
            self.prev_pend_vel = cur_pend_vel
            return 0.0
        else:
            acc = (cur_pend_vel - self.prev_pend_vel)
            self.prev_pend_vel = cur_pend_vel
            return acc

    def _reset_motor_init_count(self):
        print("calibrate motor init count...")
        push_max_count = 0
        push_min_count = 0
        push_max_cnt = 0
        push_min_cnt = 0

        while True:
            push_max_cnt += 1
            self.card.write_pwm(self.pwm_ch, 1, array('d', [-self.reset_motor_init_count_force]))  # get max push counts
            time.sleep(self.reset_motor_init_count_dt)
            ms_dt = self.reset_motor_init_count_time / self.reset_motor_init_count_dt
            if push_max_cnt > ms_dt:
                self.card.read_encoder(self.motor_enc_ch, 1, self.motor_enc_val)
                push_max_count = self.motor_enc_val[0]
                break

        while True:
            push_min_cnt += 1
            self.card.write_pwm(self.pwm_ch, 1, array('d', [self.reset_motor_init_count_force]))  # get min push counts
            time.sleep(self.reset_motor_init_count_dt)
            ms_dt = self.reset_motor_init_count_time / self.reset_motor_init_count_dt
            if push_min_cnt > ms_dt:
                self.card.read_encoder(self.motor_enc_ch, 1, self.motor_enc_val)
                push_min_count = self.motor_enc_val[0]
                break

        self.init_count = int((push_max_count + push_min_count) / 2.0)  # motor init count = (max_push_counts + min_push_counts) / 2
        print("set init count: ", self.init_count)

    def _reset_pendulum_init_count(self) -> None:
        self.card.read_encoder(self.pend_enc_ch, 1, self.pend_enc_val)
        self.pen_init_count_list.append(copy.deepcopy(self.pend_enc_val[0]))
        if len(self.pen_init_count_list) > self.reset_pendulum_init_count_mean:
            pend_init_count_mean = int(np.mean(self.pen_init_count_list))
            print("pendulum init count diff: ", (self.pend_init_count - pend_init_count_mean) % self.motor_encoder_max_count)
            self.pend_init_count = pend_init_count_mean  # pendulum init count = mean of pendulum counts

    def get_init_observations(self):
        motor_angle = self._get_motor_angle()
        pendulum_angle = self._get_pendulum_angle()
        motor_angle_vel = self._get_motor_velocity()
        pendulum_angle_vel = self._get_pendulum_velocity()
        pend_spin_num = (self.pend_init_count - self.pend_enc_val[0]) / self.motor_encoder_max_count
        observation_t = torch.tensor([motor_angle, math.sin(pendulum_angle), math.cos(pendulum_angle), motor_angle_vel, pendulum_angle_vel, pend_spin_num], dtype=torch.float32)
        return observation_t

    def noramlize_observation(self, observation):
        motor_angle_n = observation[0] / self.motor_angle_obs_max
        motor_angle_vel_n = observation[3] / self.motor_angle_vel_max
        pendulum_angle_vel_n = observation[4] / self.pendulum_angle_vel_max
        pendulum_spin_num = observation[5] / self.pendulum_spin_num_max
        observation_n = torch.tensor([motor_angle_n, observation[1], observation[2], motor_angle_vel_n, pendulum_angle_vel_n, pendulum_spin_num], dtype=torch.float32)
        return observation_n

    def reset(self):
        self.step_count = 0
        self.reset_count += 1
        self.prev_motor_angle = None
        self.prev_pend_angle = None
        self.prev_motor_vel = None
        self.prev_pend_vel = None

        print("\n======RESET START======")
        # Reset LED blue
        blue_led_values = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.card.write_other(self.led_channels, len(self.led_channels), blue_led_values)

        start = time.time()
        reset_counter = 0
        reset_success_num = 0
        motor_ang_vel = 0.0

        # reset motor init count every n resets
        if self.reset_count % self.reset_motor_init_count_interval == 0:  
            self._reset_motor_init_count()

        thresh_hold_pen_vel = self.reset_threshold_pendulum_vel
        thresh_hold_motor_ang_err = self.reset_threshold_motor_rad
        reset_done = False

        while True:
            reset_counter += 1
            cur_rad = self._get_motor_angle()
            # motor_angle_error = (target angle=0.0) - (current angle)
            motor_ang_err = -cur_rad  
            pen_vel = self._get_pendulum_velocity()

            if abs(motor_ang_err) < thresh_hold_motor_ang_err and abs(pen_vel) < thresh_hold_pen_vel:
                reset_success_num += 1
            else:
                reset_success_num = 0
                self.pen_init_count_list = []

            if reset_success_num > self.reset_success_num:
                reset_done = True

            motor_ang_vel = self._get_motor_velocity()

            # PD control
            duty = self.Kp * 2 * motor_ang_err - self.Kd * motor_ang_vel
            duty = max(min(duty, self.reset_pwm), -self.reset_pwm)

            # apply action to quanser
            self.card.write_pwm(self.pwm_ch, 1, array('d', [duty]))  
            time.sleep(self.reset_dt)

            if reset_counter > self.reset_fail_count and reset_success_num == 0:
                thresh_hold_pen_vel += 0.05
                thresh_hold_motor_ang_err += 0.05
                reset_counter = 0
                if abs(motor_ang_err) > self.reset_fail_rad:
                    self._reset_motor_init_count()
                    print("RESET FAILED, RE-CALIBRATE MOTOR INIT COUNT")
                else:
                    print("FAILED TO RESET, RETRYING... motor_ang_err: ", abs(motor_ang_err), " ,abs(pend_val): ", abs(pen_vel))
            
            if reset_done:
                self.card.write_pwm(array('I', [0]), 1, array('d', [0.0]))
                break
        
        self.pen_init_count_list = []
        # reset pendulum init count after reset
        for _ in range(1+self.reset_pendulum_init_count_mean):
            self._reset_pendulum_init_count()
            time.sleep(self.reset_pendulum_init_count_dt)

        print("\n======RESET END======")
        print(f"Reset time: {time.time() - start:.2f} sec")
        obs = self.get_init_observations()
        obs = self.noramlize_observation(obs)
        # Step LED Red
        red_led_values = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.card.write_other(self.led_channels, len(self.led_channels), red_led_values)
        return obs

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict]:
        self.step_count += 1
        terminated = False

        motor_angle = self._get_motor_angle()
        pend_angle = self._get_pendulum_angle()
        next_motor_ang_vel = self._get_motor_velocity()
        next_pend_ang_vel = self._get_pendulum_velocity()
        pend_spin_num = (self.pend_init_count - self.pend_enc_val[0]) / self.motor_encoder_max_count
        next_obs = torch.tensor([motor_angle, math.sin(pend_angle), math.cos(pend_angle), next_motor_ang_vel, next_pend_ang_vel, pend_spin_num], dtype=torch.float32)
        next_obs = self.noramlize_observation(next_obs)

        # compute reward
        reward = abs(pend_angle)**2
        reward /= math.pi**2

        if abs(pend_angle) > math.radians(self.double_reward_deg):
            reward *= 2

        reward += 0.1

        # termination conditions
        if abs(pend_spin_num) > self.terminate_pendulum_spin_num:
            print("PENDULUM SPIN OVER: ", pend_spin_num)
            terminated = True

        if abs(math.degrees(motor_angle)) > self.terminate_motor_angle_deg:
            print("MOTOR ANGLE OVER: ", math.degrees(motor_angle))
            terminated = True

        if abs(next_pend_ang_vel) > self.terminate_pendulum_angle_vel:
            print("PENDULUM VELOCITY OVER: ", next_pend_ang_vel)
            terminated = True

        # truncation: max steps reached
        truncated = self.step_count >= self.max_steps
        if truncated:
            print("TRUNCATED")
        info = {}

        if terminated or truncated:
            blue_led_values = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            self.card.write_other(self.led_channels, len(self.led_channels), blue_led_values)
            self.reset_helper()

        return next_obs, reward, terminated, truncated, info
    
    def reset_helper(self):
        reset_counter = 0
        target_angle = 0.0
        reset_success_num = 0
        motor_ang_vel = 0.0
        while True:
            reset_counter += 1
            # prevent infinite loop
            if reset_counter > self.reset_helper_max_count:
                break
            self.card.read_encoder(self.motor_enc_ch, 1, self.motor_enc_val)
            cur_rad = (self.motor_enc_val[0] - self.init_count) * 2 * math.pi / self.motor_encoder_max_count
            motor_ang_err = target_angle - cur_rad

            if abs(motor_ang_err) < self.reset_helper_threshold_motor_rad:
                reset_success_num += 1
            else:
                reset_success_num = 0

            if reset_success_num > self.reset_helper_success_num:
                self.card.write_pwm(self.pwm_ch, 1, array('d', [0.0]))
                break

            motor_ang_vel = self._get_motor_velocity()

            # PD control
            duty = self.Kp * motor_ang_err - self.Kd * motor_ang_vel

            if cur_rad > math.radians(self.reset_helper_fast_angle):
                duty = -self.reset_helper_fast_pwm
            elif cur_rad < -math.radians(self.reset_helper_fast_angle):
                duty = self.reset_helper_fast_pwm
            else:
                duty = max(min(duty, self.reset_helper_pwm), -self.reset_helper_pwm)
            
            # apply action to quanser
            self.card.write_pwm(self.pwm_ch, 1, array('d', [duty]))
            time.sleep(self.reset_helper_dt)
    

    def apply_action(self, actions):
        pwm = float(actions.item()) * self.action_scale
        pwm_buf = array('d', [pwm])
        self.card.write_pwm(self.pwm_ch, 1, pwm_buf)

    def apply_action_dqn(self, actions):
        pwm_buf = array('d', [actions])
        self.card.write_pwm(self.pwm_ch, 1, pwm_buf)        
