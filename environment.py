import numpy as np
import pandas as pd
import config
import torch
from sklearn.ensemble import IsolationForest
import random


class Environment():
    def __init__(self, dataset_a, dataset_u, dataset_test, test_label):
        self.dataset_a = dataset_a
        self.dataset_u = dataset_u
        self.dataset_test = dataset_test
        self.test_label = test_label

        self.current_data = dataset_u[random.randint(0, len(dataset_u) - 1)]
        self.current_class = 1

        self.clf = IsolationForest(contamination=config.contamination_rate)
        self.mapped = torch.tensor([])
        self.target_net = None
        self.net = None

        self.obs_dim = self.current_data.size()[0]
        self.action_dim = 2

    def reset(self):
        self.current_data = self.dataset_u[random.randint(0, len(self.dataset_u) - 1)]
        self.current_class = 1

        self.clf = IsolationForest(contamination=config.contamination_rate)
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
        self.current_data = self.dataset_a[random.randint(0, config.anomaly_num - 1)]

    def sample_method_two(self, action):
        self.current_class = 1
        candidate = np.random.choice([i for i in range(len(self.dataset_u))], size=config.sample_num, replace=False)
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
