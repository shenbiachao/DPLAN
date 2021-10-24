import config
import numpy as np
import os
from datetime import datetime
import json
from tensorboardX import SummaryWriter
import torch
import random


def soft_update_network(source_network, target_network, tau):
    for target_param, local_param in zip(target_network.parameters(),
                                        source_network.parameters()):
        target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)


def hard_update_network(source_network, target_network):
    for target_param, local_param in zip(target_network.parameters(),
                                         source_network.parameters()):
        target_param.data.copy_(local_param.data)


def second_to_time_str(remaining:int):
    dividers = [86400, 3600, 60, 1]
    names = ['day', 'hour', 'minute', 'second']
    results = []
    for d in dividers:
        re = int(np.floor(remaining / d))
        results.append(re)
        remaining -= re * d
    time_str = ""
    for re, name in zip(results, names):
        if re > 0 :
            time_str += "{} {}  ".format(re, name)
    return time_str


class Logger():
    def __init__(self, log_path, tb_dir="tb_logs", prefix="", warning_level=3, print_to_terminal=True):
        unique_path = self.make_simple_log_path(prefix)
        log_path = os.path.join(log_path, unique_path)
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.tb_writer = SummaryWriter(log_path)
        self.log_file_path = os.path.join(log_path, "output.txt")
        self.print_to_terminal = print_to_terminal
        self.warning_level = warning_level

    def make_simple_log_path(self, prefix):
        now = datetime.now()
        suffix = now.strftime("%d(%H.%M)")
        pid_str = os.getpid()
        return "{}-{}-{}".format(prefix, suffix, pid_str)

    @property
    def log_dir(self):
        return self.log_path

    def log_str(self, content, level=4):
        if level < self.warning_level:
            return
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        if self.print_to_terminal:
            print("\033[32m{}\033[0m:\t{}".format(time_str, content))
        with open(self.log_file_path, 'a+') as f:
            f.write("{}:\t{}\n".format(time_str, content))

    def log_var(self, name, val, ite):
        self.tb_writer.add_scalar(name, val, ite)

    def log_str_object(self, name: str, log_dict: dict = None, log_str: str = None):
        if log_dict is not None:
            log_str = json.dumps(log_dict, indent=4)
        elif log_str is not None:
            pass
        else:
            assert 0
        if name[-4:] != ".txt":
            name += ".txt"
        target_path = os.path.join(self.log_path, name)
        with open(target_path, 'w+') as f:
            f.write(log_str)
        self.log_str("saved {} to {}".format(name, target_path))


def set_device_and_logger(gpu_id, logger_ent):
    global device, logger
    if gpu_id < 0 or torch.cuda.is_available() == False:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(gpu_id))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    logger = logger_ent

    return device


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
