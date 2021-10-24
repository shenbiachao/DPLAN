# common
eps = 1e-9

# main

# loader
train_percentage = 0.8
contamination_rate = 0.02
ann_anomaly_class = 1
har_anomaly_class = 3
cov_anomaly_class = 6
anomaly_num = 60

# dqn_trainer
batch_size = 32
num_updates_per_iteration = 1
num_steps_per_iteration = 2000
log_interval = 1
num_test_trajectories = 1
max_iteration = 20
init_epsilon = 1
final_epsilon = 0.1
save_model_interval = 1
start_timestep = 10000
max_buffer_size = 100000
refresh_interval = 20

# dqn_agent
device = 'cuda'
update_target_network_interval = 100
gamma = 0.99
tau = 0.5
n = 1
learning_rate = 0.00025
momentum = 0.95
hidden_dims = [20]

# environment.py
sample_num = 1000
p = 0.5
