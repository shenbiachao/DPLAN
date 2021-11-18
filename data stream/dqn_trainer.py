import numpy as np
from time import time
import random
import config
from common import second_to_time_str
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
import sys


class TDReplayBuffer:
    def __init__(self, obs_dim):
        self.n = config.n
        self.max_buffer_size = config.max_buffer_size
        self.gamma = config.gamma
        action_dim = 1
        self.discrete_action = True
        self.obs_buffer = np.zeros((self.max_buffer_size, obs_dim))
        self.action_buffer = np.zeros((self.max_buffer_size, action_dim))
        self.next_obs_buffer = np.zeros((self.max_buffer_size, obs_dim))
        self.reward_buffer = np.zeros((self.max_buffer_size,))
        self.done_buffer = np.ones((self.max_buffer_size,))
        self.n_step_obs_buffer = np.zeros((self.max_buffer_size, obs_dim))
        self.discounted_reward_buffer = np.zeros((self.max_buffer_size,))
        self.n_step_done_buffer = np.zeros(self.max_buffer_size, )
        self.n_count_buffer = np.ones((self.max_buffer_size,)).astype(np.int) * self.n
        # insert a random state at initialization to avoid bugs when inserting the first state
        self.max_sample_size = 1
        self.curr = 1

    def add_tuple(self, obs, action, next_obs, reward, done):
        self.obs_buffer[self.curr] = obs
        self.action_buffer[self.curr] = action
        self.next_obs_buffer[self.curr] = next_obs
        self.reward_buffer[self.curr] = reward
        self.done_buffer[self.curr] = done

        self.n_step_obs_buffer[self.curr] = next_obs
        self.discounted_reward_buffer[self.curr] = reward
        self.n_step_done_buffer[self.curr] = 0.
        breaked = False
        for i in range(self.n - 1):
            idx = (self.curr - i - 1) % self.max_sample_size
            if self.done_buffer[idx]:
                breaked = True
                break
            self.discounted_reward_buffer[idx] += (self.gamma ** (i + 1)) * reward
        if not breaked and not self.done_buffer[(self.curr - self.n) % self.max_sample_size]:
            self.n_step_obs_buffer[(self.curr - self.n) % self.max_sample_size] = obs
        if done:
            self.n_step_done_buffer[self.curr] = 1.0
            self.n_count_buffer[self.curr] = 1
            for i in range(self.n - 1):
                idx = (self.curr - i - 1) % self.max_sample_size
                if self.done_buffer[idx]:
                    break
                self.n_step_obs_buffer[idx] = next_obs
                self.n_step_done_buffer[idx] = 1.0
                self.n_count_buffer[idx] = i + 2
        else:
            prev_idx = (self.curr - 1) % self.max_sample_size
            if not self.done_buffer[prev_idx]:
                self.n_step_done_buffer[prev_idx] = 0.
            for i in range(self.n - 1):
                idx = (self.curr - i - 1) % self.max_sample_size
                if self.done_buffer[idx]:
                    break
                self.n_step_obs_buffer[idx] = next_obs
                self.n_step_done_buffer[idx] = 0.0

        self.curr = (self.curr + 1) % self.max_buffer_size
        self.max_sample_size = min(self.max_sample_size + 1, self.max_buffer_size)

    def sample_batch(self, batch_size, to_tensor=True, n=None):
        if n is not None and n != self.n:
            self.update_td(n)
        if self.done_buffer[self.curr - 1]:
            valid_indices = range(self.max_sample_size)
        elif self.curr >= self.n:
            valid_indices = list(range(self.curr - self.n)) + list(range(self.curr + 1, self.max_sample_size))
        else:
            valid_indices = range(self.curr + 1, self.max_sample_size - (self.n - self.curr))
        batch_size = min(len(valid_indices), batch_size)
        index = random.sample(valid_indices, batch_size)
        obs_batch, action_batch, n_step_obs_batch, discounted_reward_batch, n_step_done_batch = \
            self.obs_buffer[index], \
            self.action_buffer[index], \
            self.n_step_obs_buffer[index], \
            self.discounted_reward_buffer[index], \
            self.n_step_done_buffer[index]
        if to_tensor:
            obs_batch = torch.FloatTensor(obs_batch).to(config.device)
            if self.discrete_action:
                action_batch = torch.LongTensor(action_batch).to(config.device)
            else:
                action_batch = torch.FloatTensor(action_batch).to(config.device)
            n_step_obs_batch = torch.FloatTensor(n_step_obs_batch).to(config.device)
            discounted_reward_batch = torch.FloatTensor(discounted_reward_batch).to(config.device).unsqueeze(1)
            n_step_done_batch = torch.FloatTensor(n_step_done_batch).to(config.device).unsqueeze(1)

        return obs_batch, action_batch, n_step_obs_batch, discounted_reward_batch, n_step_done_batch

    def update_td(self, n):
        print("Updating the current buffer from td \033[32m{} to {}\033[0m".format(self.n, n))
        self.n_step_obs_buffer = np.zeros_like(self.n_step_obs_buffer)
        self.discounted_reward_buffer = np.zeros_like(self.discounted_reward_buffer)
        self.n_step_done_buffer = np.zeros_like(self.n_step_done_buffer)
        self.mask_buffer = np.zeros_like(self.n_step_done_buffer)
        curr = (self.curr - 1) % self.max_sample_size
        curr_traj_end_idx = curr
        num_trajs = int(np.sum(self.done_buffer))
        if not self.done_buffer[curr]:
            num_trajs += 1
        while num_trajs > 0:
            self.n_step_done_buffer[curr_traj_end_idx] = self.done_buffer[curr_traj_end_idx]
            self.n_step_obs_buffer[curr_traj_end_idx] = self.next_obs_buffer[curr_traj_end_idx]
            curr_traj_len = 1
            idx = (curr_traj_end_idx - 1) % self.max_sample_size
            while not self.done_buffer[idx] and idx != curr:
                idx = (idx - 1) % self.max_sample_size
                curr_traj_len += 1

            for i in range(min(n - 1, curr_traj_len)):
                idx = (curr_traj_end_idx - i - 1) % self.max_sample_size
                if self.done_buffer[idx]:
                    break
                self.n_step_obs_buffer[idx] = self.next_obs_buffer[curr_traj_end_idx]
                self.n_step_done_buffer[idx] = self.done_buffer[curr_traj_end_idx]

            for i in range(curr_traj_len):
                curr_return = self.reward_buffer[(curr_traj_end_idx - i) % self.max_sample_size]
                for j in range(min(n, curr_traj_len - i)):
                    target_idx = curr_traj_end_idx - i - j
                    self.discounted_reward_buffer[target_idx] += (curr_return * (self.gamma ** j))

            if curr_traj_len >= n:
                for i in range(curr_traj_len - n):
                    curr_idx = (curr_traj_end_idx - n - i) % self.max_sample_size
                    if self.done_buffer[curr_idx]:
                        break
                    next_obs_idx = (curr_idx + n) % self.max_sample_size
                    self.n_step_obs_buffer[curr_idx] = self.obs_buffer[next_obs_idx]
            curr_traj_end_idx = (curr_traj_end_idx - curr_traj_len) % self.max_sample_size
            num_trajs -= 1

        self.n = n


class DQNTrainer:
    def __init__(self, agent, env, buffer, logger):
        self.agent = agent
        self.env = env
        self.buffer = buffer
        self.logger = logger

        self.batch_size = config.batch_size
        self.max_iteration = config.max_iteration
        self.num_steps_per_iteration = config.num_steps_per_iteration
        self.interact_update_interval = config.interact_update_interval
        self.warmup_tau = config.warmup_tau
        self.interact_tau = config.interact_tau
        self.epsilon = config.init_epsilon
        self.log_interval = config.log_interval
        self.test_interval = config.test_interval
        self.warmup_timestep = config.warmup_timestep
        self.anneal_rate = (config.init_epsilon - config.final_epsilon) / self.num_steps_per_iteration

    def warmup(self):
        iteration_durations = []
        tot_env_steps = 0
        max_acc = -float('inf')

        state = self.env.reset()
        for ite in range(1, self.max_iteration + 1):
            iteration_start_time = time()
            traj_reward = 0
            self.epsilon = config.init_epsilon

            for step in range(self.num_steps_per_iteration):
                if random.random() < self.epsilon:
                    action = random.randint(0, self.env.action_dim - 1)
                else:
                    action = self.agent.select_action(state)
                self.epsilon = self.epsilon - self.anneal_rate

                next_state, reward, done, info = self.env.step(action)
                traj_reward += reward

                self.buffer.add_tuple(state.cpu(), action, next_state.cpu(), reward, float(done))
                state = next_state

                tot_env_steps += 1
                if tot_env_steps < self.warmup_timestep:
                    continue

                data_batch = self.buffer.sample_batch(self.batch_size)
                self.agent.update(data_batch, self.warmup_tau)

                self.env.refresh_net(self.agent.q_network)

            self.env.refresh_iforest(self.agent.q_network)

            state = self.env.reset()

            iteration_end_time = time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_durations.append(iteration_duration)

            if ite % self.log_interval == 0:
                auc_roc, auc_pr, acc = self.evaluate()
                if acc > max_acc:
                    max_acc = acc
                    torch.save(self.agent.q_network.state_dict(), "best.param")

                self.logger.log_var("warmup/auc_roc", auc_roc, tot_env_steps)
                self.logger.log_var("warmup/auc_pr", auc_pr, tot_env_steps)
                self.logger.log_var("warmup/acc", acc, tot_env_steps)
                self.logger.log_var("warmup/reward", traj_reward, tot_env_steps)
                remaining_seconds = int((self.max_iteration - ite) * np.mean(iteration_durations[-3:]))
                time_remaining_str = second_to_time_str(remaining_seconds)
                summary_str = "iteration {}/{}:\ttrain return {:.02f}\t" \
                              "auc_roc {:02f}\tauc_pr {:02f}\tacc {:02f}\twarmup eta: {}".format(
                    ite, self.max_iteration, traj_reward, auc_roc, auc_pr, acc, time_remaining_str)
                self.logger.log_str(summary_str)

    def interact(self):
        self.agent.q_network.load_state_dict(torch.load("best.param"))
        self.epsilon = config.lower_bound
        prediction_history = []
        tot_env_steps = 0
        state = self.env.reset()
        self.epsilon = config.final_epsilon

        while True:
            flag, nex, label = self.env.dataset_step()
            tot_env_steps += 1
            if not flag:
                self.logger.log_str("No more data available.")
                sys.exit(0)

            _, pred = torch.max(self.agent.q_network(nex), dim=1)
            if pred.cpu().detach().numpy()[0] == label:
                prediction_history.append(1)
            else:
                prediction_history.append(0)

            if tot_env_steps % self.interact_update_interval == 0:
                for step in range(self.num_steps_per_iteration):
                    if random.random() < self.epsilon:
                        action = random.randint(0, self.env.action_dim - 1)
                    else:
                        action = self.agent.select_action(state)

                    next_state, reward, done, info = self.env.step(action)

                    self.buffer.add_tuple(state.cpu(), action, next_state.cpu(), reward, float(done))
                    state = next_state

                    data_batch = self.buffer.sample_batch(self.batch_size)
                    self.agent.update(data_batch, self.interact_tau)

                    self.env.refresh_net(self.agent.q_network)

                self.env.refresh_iforest(self.agent.q_network)
                state = self.env.reset()

            if tot_env_steps % self.test_interval == 0:
                auc_roc, auc_pr, acc = self.evaluate()
                real_time_acc = np.mean(prediction_history[-100:])
                self.logger.log_var("interact/auc_roc", auc_roc, tot_env_steps)
                self.logger.log_var("interact/auc_pr", auc_pr, tot_env_steps)
                self.logger.log_var("interact/acc", acc, tot_env_steps)
                self.logger.log_var("interact/real_time_acc", real_time_acc, tot_env_steps)
                summary_str = "auc_roc {:02f}\tauc_pr {:02f}\tacc {:02f}\trecent real_time_acc {:02f}\t".format(
                    auc_roc, auc_pr, acc, real_time_acc)
                self.logger.log_str(summary_str)

    def evaluate(self):
        if not self.env.refresh_dataset_test():
            self.logger.log_str("No more test data available.")
            sys.exit(0)
        q_values = self.agent.q_network(self.env.dataset_test)
        anomaly_score = q_values[:, 1]
        _, action_indices = torch.max(q_values, dim=1)
        auc_roc = roc_auc_score(self.env.test_label, anomaly_score.cpu().detach())
        precision, recall, _thresholds = precision_recall_curve(self.env.test_label, anomaly_score.cpu().detach())
        auc_pr = auc(recall, precision)
        acc = accuracy_score(self.env.test_label, action_indices.cpu().detach())

        return auc_roc, auc_pr, acc
