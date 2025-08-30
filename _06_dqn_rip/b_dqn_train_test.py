import collections
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import gymnasium as gym
from collections import deque


import os
import time
from datetime import datetime
from shutil import copyfile
import wandb

from _06_dqn_rip.c_qnet import QCritic, MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISCRETE_PWMS = [-0.35, -0.02, 0.0, 0.02, 0.35]

Transition = collections.namedtuple(
    typename="Transition", field_names=["observation", "action", "next_observation", "reward", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def size(self) -> int:
        return len(self.buffer)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def pop(self) -> Transition:
        return self.buffer.pop()

    def clear(self) -> None:
        self.buffer.clear()

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get random index
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        # Sample
        observations, actions, next_observations, rewards, dones = zip(*[self.buffer[idx] for idx in indices])

        # Convert to ndarray for speed up cuda
        observations = np.array(observations)
        next_observations = np.array(next_observations)

        actions = np.array(actions)
        actions = np.expand_dims(actions, axis=-1) if actions.ndim == 1 else actions
        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards
        dones = np.array(dones, dtype=bool)

        # Convert to tensor
        observations = torch.tensor(observations, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

        return observations, actions, next_observations, rewards, dones
    

class DQNTrainer:
    def __init__(self, env: gym.Env, config: dict, use_wandb: bool):
        self.env = env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]
        custom_name = "DQN"
        self.current_time = custom_name + datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S_")

        if use_wandb:
            self.wandb = wandb.init(project="DDPG_{0}".format(self.env_name), name=self.current_time, config=config)
        else:
            self.wandb = None

        self.max_num_episodes = config["max_num_episodes"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.print_episode_interval = config["print_episode_interval"]
        self.episode_reward_avg_solved = config["episode_reward_avg_solved"]
        self.soft_update_tau = config["soft_update_tau"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.epsilon_start = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.training_start_steps = config["training_start_steps"]
        self.epsilon_final_scehduled_percent = config["epsilon_final_scehduled_percent"]
        self.forward_start_episode = config["forward_start_episode"]
        self.train_per_step = config["train_per_step"]
        self.step_control_time = config["step_control_time"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.clip_grad_max_norm = config["clip_grad_max_norm"]
        self.valiation_interval = config["valiation_interval"]
        self.validation_start_episode = config["validation_start_episode"]
        self.validation_start_gap_reward = config["validation_start_gap_reward"]

        self.q_critic = QCritic(n_features=env.observation_space.shape[0], n_actions=len(DISCRETE_PWMS))
        self.target_q_critic = QCritic(n_features=env.observation_space.shape[0], n_actions=len(DISCRETE_PWMS))
        self.target_q_critic.load_state_dict(self.q_critic.state_dict())
        self.q_critic_optimizer = optim.Adam(self.q_critic.parameters(), lr=self.learning_rate)


        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_size)

        self.epsilon_scheduled_last_episode = self.max_num_episodes * self.epsilon_final_scehduled_percent

        self.time_steps = 0
        self.training_time_steps = 0
        self.epsilon_steps = 0


        self.total_train_start_time = None
        self.train_start_time = None
        self.step_call_time = None
  

    def epsilon_scheduled(self, current_episode: int) -> float:
        if current_episode < self.forward_start_episode:
            return self.epsilon_start
        fraction = min((current_episode - self.forward_start_episode) / self.epsilon_scheduled_last_episode, 1.0)
        epsilon_span = self.epsilon_start - self.epsilon_end

        epsilon = min(self.epsilon_start - fraction * epsilon_span, self.epsilon_start)
        return epsilon

    def train_loop(self) -> None:
        self.total_train_start_time = time.time()
        is_terminated = False

        for n_episode in range(1, self.max_num_episodes + 1):
            loss_list = []
            episode_reward = 0.0
            episode_steps = 0
            done = False
            step_call_time_deque = deque(maxlen=2000)
            action_list = deque(maxlen=2000)
            self.step_call_time = None
            epsilon = self.epsilon_scheduled(n_episode)

            observation = self.env.reset()
            print("======EPISODE START====== ")
            episode_start_time = time.time()
            while not done:
                self.train_start_time = time.time()
                self.time_steps += 1

                self.epsilon_steps += 1
                if n_episode < self.forward_start_episode:
                    action = self.q_critic.get_action_eps(observation, 1.0) # random action
                else:

                    action = self.q_critic.get_action_eps(observation, epsilon) # epsilon-greedy

                action_list.append(DISCRETE_PWMS[action])

                self.env.apply_action_dqn(DISCRETE_PWMS[action])  # apply action to quanser

                if self.time_steps > self.training_start_steps:
                    for _ in range(self.train_per_step):
                        loss = self.train()
                        loss_list.append(loss)
                
                if self.step_call_time is not None:
                    while (time.perf_counter() - self.step_call_time) < self.step_control_time:
                        time.sleep(0.0001)
                    step_call_time_deque.append(time.perf_counter() - self.step_call_time - self.step_control_time)
                    
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                self.step_call_time = time.perf_counter()

                episode_reward += reward

                transition = Transition(observation, action, next_observation, reward, terminated)

                self.replay_buffer.append(transition)

                observation = next_observation
                done = terminated or truncated

                episode_steps += 1
                

            print("======EPISODE END====== ")
            episode_time = time.time() - episode_start_time
            if len(step_call_time_deque) > 0:
                step_call_time_deque.pop()
                print(f"STEP CALL TIME ERROR MEAN: {np.mean(step_call_time_deque):.6f} sec")

            if episode_reward > self.episode_reward_avg_solved - self.validation_start_gap_reward:
                is_terminated = self.validate()

            if n_episode % self.valiation_interval == 0 and n_episode > self.validation_start_episode:
                is_terminated = self.validate()

            loss_mean = np.mean(loss_list)


            if n_episode % self.print_episode_interval == 0:
                print(
                    "\n[Episode {:3,}, Time Steps {:6,}]".format(n_episode, self.time_steps),
                    "\nEpisode Reward: {:>9.3f},".format(episode_reward),
                    "\nLoss: {:>7.3f},".format(loss_mean),
                    "\nTraining Steps: {:5,}, ".format(self.training_time_steps),
                    "\nEpisode Time: {},".format(time.strftime("%H:%M:%S", time.gmtime(episode_time))),
                    "\nEpsilon: {:>7.3f}, \n".format(epsilon),
                )
                print("####################################################################################")

            if self.use_wandb:
                self.log_wandb(
                    episode_reward,
                    loss_mean,
                    n_episode,
                    epsilon,
                    action_list,
                    self.time_steps
                )

            if is_terminated:
                break

        total_training_time = time.time() - self.total_train_start_time
        total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))
        if self.use_wandb:
            self.wandb.finish()

    def log_wandb(
        self,
        episode_reward: float,
        loss: float,
        n_episode: float,
        epsilon: float,
        action_list: deque,
        total_steps: int
    ) -> None:
        self.wandb.log(
            {
                "[TRAIN] Episode Reward": episode_reward,
                "[TRAIN] Loss": loss,
                "[TRAIN] Replay buffer": self.replay_buffer.size(),
                "[TRAIN] Epsilon": epsilon,
                "[TRAIN] EPISODE ACTION MEAN": np.mean(action_list),
                "[TRAIN] Total Steps": total_steps,
                "Training Episode": n_episode,
                "Training Steps": self.training_time_steps,
            }
        )
    
    def train(self) -> tuple[float, float, float]:
        self.training_time_steps += 1
        observations, actions, next_observations, rewards, dones = self.replay_buffer.sample(self.batch_size)

        q_values = self.q_critic(observations)
        q_value = q_values.gather(dim=1, index=actions)
        with torch.no_grad():
            q_values_next = self.target_q_critic(next_observations)
            max_vals= q_values_next.max(dim=1, keepdim=True).values
            max_vals[dones] = 0.0
            target = rewards + self.gamma * max_vals
        
        loss = F.mse_loss(q_value, target)

        self.q_critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_critic.parameters(), self.clip_grad_max_norm)
        self.q_critic_optimizer.step()

        self.soft_synchronize_models(source_model=self.q_critic, target_model=self.target_q_critic, tau=self.soft_update_tau)

        return loss.item()
    
    def soft_synchronize_models(self, source_model, target_model, tau) -> None:
        source_model_state = source_model.state_dict()
        target_model_state = target_model.state_dict()
        for k, v in source_model_state.items():
            target_model_state[k] = (1.0 - tau) * target_model_state[k] + tau * v
        target_model.load_state_dict(target_model_state)

    def model_save(self, val_reward_avg: float) -> None:
        os.makedirs(MODEL_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"dqn_{self.env_name}_{val_reward_avg:.1f}_qnet_{timestamp}.pth"
        filepath  = os.path.join(MODEL_DIR, filename)

        torch.save(self.q_critic.state_dict(), filepath)

        copyfile(filepath, os.path.join(MODEL_DIR, f"dqn_{self.env_name}_qnet_latest.pth"))


    def validate(self) -> bool:
        is_terminated = False
        validation_episode_reward_list = []

        for i in range(self.validation_num_episodes):
            validation_episode_reward = 0
            observation = self.env.reset()
            done = False
            print("***********VALIDATION START*********** ")
            while not done:

                action = self.q_critic.get_action_eps(observation, 0.0)

                self.env.apply_action_dqn(DISCRETE_PWMS[action])  # apply action to quanser

                time.sleep(self.step_control_time)

                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                validation_episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            validation_episode_reward_list.append(validation_episode_reward)
            print(f"Validation {i+1} reward: {validation_episode_reward}")

        validation_episode_reward_mean = np.mean(validation_episode_reward_list)
        print(f"[Validation Episode Reward Mean: {validation_episode_reward_mean}]")
        if self.use_wandb:
            self.wandb.log({"[VALIDATION] Episode Reward": validation_episode_reward_mean})

        if validation_episode_reward_mean > self.episode_reward_avg_solved:
            print("Solved in {0:,} time steps ({1:,} training steps)!".format(self.time_steps, self.training_time_steps))
            self.model_save(validation_episode_reward_mean)
            is_terminated = True
        return is_terminated
    

class DQNTester:
    def __init__(self, env, num_episodes, model_path):
        self.env = env
        self.qnet = QCritic(n_features=env.observation_space.shape[0], n_actions=len(DISCRETE_PWMS))
        self.num_episodes = num_episodes

        model_params = torch.load(model_path, map_location=DEVICE)
        self.qnet.load_state_dict(model_params)
        self.qnet.to(DEVICE)
        self.qnet.eval()

    def test(self) -> None:
        for i in range(self.num_episodes):
            episode_reward = 0

            observation = self.env.reset()

            episode_steps = 0

            done = False

            while not done:
                episode_steps += 1
                action = self.qnet.get_action_eps(observation, 0.0)
                self.env.apply_action_dqn(DISCRETE_PWMS[action])  # apply action to quanser
                time.sleep(0.006)

                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(i, episode_steps, episode_reward))