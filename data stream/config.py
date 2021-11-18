from common import try_gpu

# common
eps = 1e-9

# main

# loader
contamination_rate = 0.04
ann_anomaly_class = 1

# dqn_trainer
batch_size = 32
num_steps_per_iteration = 2000
interact_update_interval = 500
update_target_network_interval = 2000
log_interval = 1
max_iteration = 50
init_epsilon = 1
final_epsilon = 0.1
warmup_timestep = 10000
max_buffer_size = 100000
test_interval = 10
test_size = 500

# dqn_agent
device = try_gpu(0)
gamma = 0.99
warmup_tau = 0.5
interact_tau = 0.95
n = 1
learning_rate = 0.002
hidden_dims = [10]

# environment.py
sample_size = 1000
p = 0.5
init_size_rate = 0.4
know_rate = 0.6
upper_bound = 1.5
lower_bound = -0.5
