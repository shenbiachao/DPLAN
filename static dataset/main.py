import loader
from dqn_agent import DQNAgent
from dqn_trainer import DQNTrainer
import click
from common import Logger
from common import set_device_and_logger, set_global_seed
from dqn_trainer import TDReplayBuffer
from environment import Environment


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--log-dir", default="logs")
@click.option("--gpu", type=int, default=0)
@click.option("--print-log", type=bool, default=True)
@click.option("--seed", type=int, default=3)
@click.option("--info", type=str, default="")
def main(log_dir, gpu, print_log, seed, info):
    # set global seed
    set_global_seed(seed)

    # initialize logger
    env_name = "Anomaly_dectection"
    logger = Logger(log_dir, prefix=env_name + "-" + info, print_to_terminal=print_log)
    logger.log_str("logging to {}".format(logger.log_path))

    # set device and logger
    dev = set_device_and_logger(gpu, logger)
    logger.log_str("Setting device: " + str(dev))

    # load data
    logger.log_str("Loading Data")
    dataset_a, dataset_u, dataset_test, test_label = loader.load_ann()

    # initialize environment
    logger.log_str("Initializing Environment")

    env = Environment(dataset_a, dataset_u, dataset_test, test_label)
    eval_env = Environment(dataset_a, dataset_u, dataset_test, test_label)

    # initialize buffer
    logger.log_str("Initializing Buffer")
    buffer = TDReplayBuffer(env.obs_dim)

    # initialize model
    logger.log_str("Initializing Agent")
    agent = DQNAgent(env.obs_dim, env.action_dim)

    env.refresh_net(agent.q_network)
    eval_env.refresh_net(agent.q_network)
    env.refresh_iforest(agent.q_network)
    eval_env.refresh_iforest(agent.q_network)

    # initialize trainer
    logger.log_str("Initializing Trainer")
    trainer = DQNTrainer(agent, env, eval_env, buffer, logger)

    logger.log_str("Starting Training")

    trainer.train()


if __name__ == '__main__':
    main()
