import os
import collections
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as tnn_utils
import numpy as np
import time
import gymnasium as gym
import wandb

from datetime import datetime
from collections import deque
from shutil import copyfile

from _07_ddpg_rip.b_actor_critic_network import Actor, QCritic, MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(precision=5, suppress=True)

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
        actions = torch.tensor(actions, dtype=torch.float32, device=DEVICE)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

        return observations, actions, next_observations, rewards, dones
    
    
class LinearNoiseScheduler:
    def __init__(self, start_scale: float, end_scale: float, decay_steps: int):
        self.start = start_scale
        self.end = end_scale
        self.decay_steps = decay_steps

    def get_scale(self, step: int) -> float:
        if step >= self.decay_steps:
            return self.end
        frac = step / self.decay_steps
        return self.start + frac * (self.end - self.start)


class DDPGTrainer:
    def __init__(self, env: gym.Env, config: dict, use_wandb: bool):
        self.env = env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]
        custom_name = "ddpg"
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
        self.training_start_steps = config["training_start_steps"]
        self.forward_start_episode = config["forward_start_episode"]
        self.train_per_step = config["train_per_step"]
        self.step_control_time = config["step_control_time"]
        self.clip_grad_max_norm = config["clip_grad_max_norm"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.valiation_interval = config["valiation_interval"]
        self.validation_start_episode = config["validation_start_episode"]
        self.validation_start_gap_reward = config["validation_start_gap_reward"]
        self.start_noise_scale = config["start_noise_scale"]
        self.end_noise_scale = config["end_noise_scale"]
        self.noise_decay_steps = config["noise_decay_steps"]
        self.noise_keep_steps = config["noise_keep_steps"]
        self.policy_update_interval = config["policy_update_interval"]
        self.target_policy_noise = config["target_policy_noise"]
        self.target_noise_clip = config["target_noise_clip"]

        self.actor = Actor(n_features=env.observation_space.shape[0], n_actions=env.action_space.shape[0])
        self.target_actor = Actor(n_features=env.observation_space.shape[0], n_actions=env.action_space.shape[0])
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.q_critic = QCritic(n_features=env.observation_space.shape[0], n_actions=env.action_space.shape[0])
        self.target_q_critic = QCritic(n_features=env.observation_space.shape[0], n_actions=env.action_space.shape[0])
        self.target_q_critic.load_state_dict(self.q_critic.state_dict())
        self.q_critic_optimizer = optim.Adam(self.q_critic.parameters(), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_size)

        self.time_steps = 0
        self.training_time_steps = 0

        self.total_train_start_time = None
        self.train_start_time = None

        self.step_call_time = None

        self.nosie_scale_scheduler = LinearNoiseScheduler(
            start_scale=self.start_noise_scale,
            end_scale=self.end_noise_scale,
            decay_steps=self.noise_decay_steps
        )

    def train_loop(self) -> None:
        self.total_train_start_time = time.time()
        policy_loss = critic_loss = mu_v = 0.0
        is_terminated = False

        for n_episode in range(1, self.max_num_episodes + 1):
            policy_loss_list = []
            critic_loss_list = []
            mu_v_list = []
            episode_reward = 0
            episode_steps = 0
            done = False
            step_call_time_deque = deque(maxlen=2000)
            action_list = deque(maxlen=2000)
            self.step_call_time = None

            observation = self.env.reset()
            print("======EPISODE START====== ")
            episode_start_time = time.time()
            while not done:
                self.train_start_time = time.time()
                self.time_steps += 1
                if n_episode < self.forward_start_episode:
                    scale = 2.0
                else:
                    scale = self.nosie_scale_scheduler.get_scale(self.time_steps)
                action = self.actor.get_action(observation, scale=scale, exploration=False)
                if episode_steps % self.noise_keep_steps == 0:
                    noise = np.random.normal(size=1, loc=0.0, scale=scale)
                action += noise
                action = np.clip(action, a_min=-1.0, a_max=1.0)
                action_list.append(action.item())

                self.env.apply_action(action)  # apply action to quanser

                if self.time_steps > self.training_start_steps:
                    for _ in range(self.train_per_step):
                        policy_loss, critic_loss, mu_v = self.train()
                        if policy_loss is not None:
                            policy_loss_list.append(policy_loss)
                            mu_v_list.append(mu_v)
                        critic_loss_list.append(critic_loss)
                
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

            policy_loss_mean = np.mean(policy_loss_list)
            critic_loss_mean = np.mean(critic_loss_list)
            mu_v_mean = np.mean(mu_v_list)

            if n_episode % self.print_episode_interval == 0:
                print(
                    "\n[Episode {:3,}, Time Steps {:6,}]".format(n_episode, self.time_steps),
                    "\nEpisode Reward: {:>9.3f},".format(episode_reward),
                    "\nmu_v Loss: {:>7.3f},".format(policy_loss_mean),
                    "\nCritic Loss: {:>7.3f},".format(critic_loss_mean),
                    "\nTraining Steps: {:5,}, ".format(self.training_time_steps),
                    "\nEpisode Time: {}, \n".format(time.strftime("%H:%M:%S", time.gmtime(episode_time))),
                )
                print("####################################################################################")

            if self.use_wandb:
                self.log_wandb(
                    episode_reward,
                    policy_loss_mean,
                    critic_loss_mean,
                    mu_v_mean,
                    n_episode,
                    scale,
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
        policy_loss: float,
        critic_loss: float,
        mu_v: float,
        n_episode: float,
        scale: float,
        action_list: deque,
        total_steps: int
    ) -> None:
        self.wandb.log(
            {
                "[TRAIN] Episode Reward": episode_reward,
                "[TRAIN] Policy Loss": policy_loss,
                "[TRAIN] Critic Loss": critic_loss,
                "[TRAIN] Batch mu_v": mu_v,
                "[TRAIN] Replay buffer": self.replay_buffer.size(),
                "[TRAIN] Noise Scale": scale,
                "[TRAIN] EPISODE ACTION MEAN": np.mean(action_list),
                "[TRAIN] Total Steps": total_steps,
                "Training Episode": n_episode,
                "Training Steps": self.training_time_steps,
            }
        )

    def train(self) -> tuple[float, float, float]:
        self.training_time_steps += 1
        observations, actions, next_observations, rewards, dones = self.replay_buffer.sample(self.batch_size)

        # CRITIC UPDATE
        q_values = self.q_critic(observations, actions)
        with torch.no_grad():
            next_mu_v = self.target_actor(next_observations)
            noise = (torch.randn_like(next_mu_v) * self.target_policy_noise).clamp_(-self.target_noise_clip, self.target_noise_clip)
            next_mu_v = (next_mu_v + noise).clamp_(-1.0, 1.0)
            next_q_values = self.target_q_critic(next_observations, next_mu_v)
            next_q_values[dones] = 0.0
            target_values = rewards + self.gamma * next_q_values

        critic_loss = F.mse_loss(q_values, target_values)

        self.q_critic_optimizer.zero_grad()
        critic_loss.backward()
        tnn_utils.clip_grad_norm_(self.q_critic.parameters(), max_norm=self.clip_grad_max_norm)
        self.q_critic_optimizer.step()
        
        actor_loss_val = None
        mu_v_mean_val = None

        if self.training_time_steps % self.policy_update_interval == 0:
        # ACTOR UPDATE
            mu_v = self.actor(observations)
            q_v = self.q_critic(observations, mu_v)
            actor_objective = q_v.mean()
            actor_loss = -1.0 * actor_objective

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            tnn_utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.clip_grad_max_norm)
            self.actor_optimizer.step()

            actor_loss_val = actor_loss.item()
            mu_v_mean_val = mu_v.mean().item()
            
        self.soft_synchronize_models(
            source_model=self.q_critic, target_model=self.target_q_critic, tau=self.soft_update_tau
        )
        self.soft_synchronize_models(
            source_model=self.actor, target_model=self.target_actor, tau=self.soft_update_tau
        )

        return actor_loss_val, critic_loss.item(), mu_v_mean_val

    def soft_synchronize_models(self, source_model, target_model, tau) -> None:
        source_model_state = source_model.state_dict()
        target_model_state = target_model.state_dict()
        for k, v in source_model_state.items():
            target_model_state[k] = (1.0 - tau) * target_model_state[k] + tau * v
        target_model.load_state_dict(target_model_state)

    def model_save(self, val_reward_avg: float) -> None:
        os.makedirs(MODEL_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"ddpg_{self.env_name}_{val_reward_avg:.1f}_actor_{timestamp}.pth"
        filepath  = os.path.join(MODEL_DIR, filename)

        torch.save(self.actor.state_dict(), filepath)

        copyfile(filepath, os.path.join(MODEL_DIR, f"ddpg_{self.env_name}_actor_latest.pth"))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"ddpg_{self.env_name}_{val_reward_avg:.1f}_critic_{timestamp}.pth"
        filepath  = os.path.join(MODEL_DIR, filename)

        torch.save(self.q_critic.state_dict(), filepath)

        copyfile(filepath, os.path.join(MODEL_DIR, f"ddpg_{self.env_name}_critic_latest.pth"))

    def validate(self) -> bool:
        is_terminated = False
        validation_episode_reward_list = []

        for i in range(self.validation_num_episodes):
            validation_episode_reward = 0
            observation = self.env.reset()
            done = False
            print("***********VALIDATION START*********** ")
            while not done:
                action = self.actor.get_action(observation, exploration=False)

                self.env.apply_action(action)  # apply action to quanser

                time.sleep(self.step_control_time)

                observation, reward, terminated, truncated, _ = self.env.step(action)

                validation_episode_reward += reward

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


class DDPGTester:
    def __init__(self, env, num_episodes, model_path):
        self.env = env
        self.actor = Actor(n_features=env.observation_space.shape[0], n_actions=env.action_space.shape[0])
        self.num_episodes = num_episodes

        model_params = torch.load(model_path, map_location=DEVICE)
        self.actor.load_state_dict(model_params)
        self.actor.to(DEVICE)
        self.actor.eval()

    def test(self) -> None:
        for i in range(self.num_episodes):
            episode_reward = 0

            observation = self.env.reset()

            episode_steps = 0

            done = False

            while not done:
                episode_steps += 1
                action = self.actor.get_action(observation, exploration=False)
                self.env.apply_action(action)  # apply action to quanser
                time.sleep(0.006)

                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(i, episode_steps, episode_reward))
