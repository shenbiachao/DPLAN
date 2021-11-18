import numpy as np
import random
import config
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
from sklearn.ensemble import IsolationForest
import sys


class IforestTrainer:
    def __init__(self, env, logger):
        self.clf = IsolationForest(contamination=config.contamination_rate)
        self.env = env
        self.logger = logger

        self.interact_update_interval = config.interact_update_interval
        self.test_interval = config.test_interval

    def warmup(self):
        dataset = torch.cat([self.env.dataset_a, self.env.dataset_u])
        self.clf.fit(dataset.cpu())

    def interact(self):
        prediction_history = []
        tot_env_steps = 0
        while True:
            flag, nex, label = self.env.dataset_step()
            tot_env_steps += 1
            if not flag:
                self.logger.log_str("No more data available.")
                sys.exit(0)

            pred = self.clf.predict(nex.cpu())

            if (pred == 1 and label == 0) or (pred == -1 and label == 1):
                prediction_history.append(1)
            else:
                prediction_history.append(0)

            if tot_env_steps % self.interact_update_interval == 0:
                dataset = torch.cat([self.env.dataset_a, self.env.dataset_u])
                self.clf.fit(dataset.cpu())

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
        anomaly_score = -self.clf.score_samples(self.env.dataset_test.cpu())
        pred_label = self.clf.predict(self.env.dataset_test.cpu())
        pred_label = [0 if i == 1 else 1 for i in pred_label]
        auc_roc = roc_auc_score(self.env.test_label, anomaly_score)
        precision, recall, _thresholds = precision_recall_curve(self.env.test_label, anomaly_score)
        auc_pr = auc(recall, precision)
        acc = accuracy_score(self.env.test_label, pred_label)

        return auc_roc, auc_pr, acc
