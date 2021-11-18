import numpy as np
import pandas as pd
import config
import torch
from sklearn.ensemble import IsolationForest
import random


class Environment():
    def __init__(self, dataset_a1, dataset_a2, dataset_n):
        self.dataset_a1 = dataset_a1
        self.dataset_a2 = dataset_a2
        self.dataset_n = dataset_n

        self.dataset_a = dataset_a1[0: 1]
        self.dataset_u = dataset_n[0: 1]
        self.available_a1_index = [i for i in range(1, len(dataset_a1))]
        self.available_a2_index = [i for i in range(0, len(dataset_a2))]
        self.available_n_index = [i for i in range(1, len(dataset_n))]
        self.dataset_test = torch.tensor([])
        self.test_label = torch.tensor([])
        self.a1_percentage = config.upper_bound
        self.initialize()
        self.refresh_dataset_test()

        self.current_data = self.dataset_u[random.randint(0, len(self.dataset_u) - 1)]
        self.current_class = 1

        self.clf = IsolationForest(contamination=config.contamination_rate * (1 - config.know_rate))
        self.mapped = torch.tensor([])
        self.target_net = None
        self.net = None

        self.obs_dim = self.current_data.size()[0]
        self.action_dim = 2

    def initialize(self):
        while len(self.dataset_u) / len(self.dataset_n) < 0.5:
            self.dataset_step()

    def dataset_step(self):
        flag = False
        add = torch.tensor([])
        label = -1
        percentage = self.a1_percentage
        percentage = max(min(1, percentage), 0)

        if np.random.choice([True, False], p=[config.contamination_rate, 1 - config.contamination_rate]):
            if np.random.choice([True, False], p=[percentage, 1 - percentage]) and len(self.available_a1_index) > 0:
                index = np.random.choice(self.available_a1_index)
                self.available_a1_index.remove(index)
                add = self.dataset_a1[index: index+1]
                label = 1
                if np.random.choice([True, False], p=[config.know_rate, 1 - config.know_rate]):
                    self.dataset_a = torch.cat([self.dataset_a, add])
                else:
                    self.dataset_u = torch.cat([self.dataset_u, add])
                flag = True
            elif len(self.available_a2_index) > 0:
                index = np.random.choice(self.available_a2_index)
                self.available_a2_index.remove(index)
                add = self.dataset_a2[index: index+1]
                label = 1
                if np.random.choice([True, False], p=[config.know_rate, 1 - config.know_rate]):
                    self.dataset_a = torch.cat([self.dataset_a, add])
                else:
                    self.dataset_u = torch.cat([self.dataset_u, add])
                flag = True
        elif len(self.available_n_index) > 0:
            index = np.random.choice(self.available_n_index)
            self.available_n_index.remove(index)
            add = self.dataset_n[index: index+1]
            label = 0
            self.dataset_u = torch.cat([self.dataset_u, add])
            flag = True

        self.a1_percentage = self.a1_percentage - (config.upper_bound - config.lower_bound) / len(self.dataset_n)
        return flag, add, label

    def refresh_dataset_test(self):
        if int(config.test_size * (1 - config.contamination_rate)) > len(self.available_n_index) \
                or int(config.test_size * config.contamination_rate * self.a1_percentage) > len(self.available_a1_index) \
                or int(config.test_size * config.contamination_rate * (1 - self.a1_percentage)) > len(self.available_a2_index):
            return False

        index_n = np.random.choice(self.available_n_index,
                                     size=int(config.test_size * (1 - config.contamination_rate)), replace=False)
        temp_n = self.dataset_n[index_n]
        index_a1 = np.random.choice(self.available_a1_index,
                                   size=int(config.test_size * config.contamination_rate * self.a1_percentage), replace=False)
        temp_a1 = self.dataset_a1[index_a1]
        index_a2 = np.random.choice(self.available_a2_index,
                                    size=int(config.test_size * config.contamination_rate * (1 - self.a1_percentage)), replace=False)
        temp_a2 = self.dataset_a2[index_a2]

        self.dataset_test = torch.cat([temp_n, temp_a1, temp_a2])
        self.test_label = torch.tensor([0] * len(temp_n) + [1] * (len(temp_a1) + len(temp_a2)))
        return True

    def reset(self):
        self.current_data = self.dataset_u[random.randint(0, len(self.dataset_u) - 1)]
        self.current_class = 1

        self.clf = IsolationForest(contamination=config.contamination_rate * (1 - config.know_rate))
        self.mapped = torch.tensor([])
        self.refresh_iforest(self.net)

        return self.current_data

    def refresh_net(self, net):
        self.net = net

    def refresh_iforest(self, net):
        self.target_net = net
        with torch.no_grad():
            self.mapped = net.map(self.dataset_u).cpu()
        self.clf.fit(self.mapped)

    def intrinsic_reward(self):
        target = self.target_net.map(self.current_data)
        score = -self.clf.score_samples(target.detach().cpu().numpy().reshape(1, -1))

        return score

    def external_reward(self, action):
        if self.current_class == 0 and action == 1:
            score = 1
        elif self.current_class == 1 and action == 0:
            score = 0
        else:
            score = -1

        return score

    def sample_method_one(self):
        self.current_class = 0
        self.current_data = self.dataset_a[random.randint(0, len(self.dataset_a) - 1)]

    def sample_method_two(self, action):
        self.current_class = 1
        candidate = np.random.choice([i for i in range(len(self.dataset_u))], size=config.sample_size, replace=False)
        with torch.no_grad():
            mapped_current = self.net.map(self.current_data).cpu()
        if action == 0:
            max_dist = -float('inf')
            for ind in candidate:
                dist = np.linalg.norm(mapped_current - self.net.map(self.dataset_u[ind]).detach().cpu())
                if dist > max_dist:
                    max_dist = dist
                    self.current_data = self.dataset_u[ind]
        else:
            min_dist = float('inf')
            for ind in candidate:
                dist = np.linalg.norm(mapped_current - self.net.map(self.dataset_u[ind]).detach().cpu())
                if dist < min_dist and dist != 0:
                    min_dist = dist
                    self.current_data = self.dataset_u[ind]

    def step(self, action):
        r_i = self.intrinsic_reward()[0]
        r_e = self.external_reward(action)
        reward = r_i + r_e

        choice = np.random.choice([0, 1], size=1, p=[config.p, 1 - config.p])
        if choice == 0:
            self.sample_method_one()
        else:
            self.sample_method_two(action)

        return self.current_data, reward, False, " "
