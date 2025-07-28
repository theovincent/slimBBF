import os
import sys

import jax
import numpy as np

from experiments.base.bbf import train
from experiments.base.utils import prepare_logs
from slimdqn.environments.atari import AtariEnv
from slimdqn.networks.bbf import BBF
from slimdqn.sample_collection.subseq_replay_buffer import SubsequenceReplayBuffer
from slimdqn.sample_collection.samplers import PrioritizedSamplingDistribution


def run(argvs=sys.argv[1:]):
    env_name, algo_name = os.path.abspath(__file__).split("/")[-2], os.path.abspath(__file__).split("/")[-1][:-3]
    p = prepare_logs(env_name, algo_name, argvs)

    q_key, train_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = AtariEnv(p["experiment_name"].split("_")[-1], sticky_actions=False)
    rb = SubsequenceReplayBuffer(
        sampling_distribution=PrioritizedSamplingDistribution(p["seed"], p["replay_buffer_capacity"]),
        max_capacity=p["replay_buffer_capacity"],
        batch_size=p["batch_size"],
        observation_shape=(env.state_height, env.state_width),
        observation_dtype=np.uint8,
        spr_window=p["spr_jumps"],
        stack_size=4,
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
        clipping=lambda x: np.clip(x, -1, 1),
    )
    agent = BBF(
        q_key,
        (env.state_height, env.state_width, env.n_stacked_frames),
        env.n_actions,
        n_bins=p["n_bins"],
        features=p["features"],
        learning_rate=p["learning_rate"],
        min_gamma=p["min_gamma"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        max_update_horizon=p["max_update_horizon"],
        horizon_cycle_steps=p["horizon_cycle_steps"],
        update_to_data=p["update_to_data"],
        n_updates_per_train_step=p["n_updates_per_train_step"],
        target_update_tau=p["target_update_tau"],
        reset_frequency=p["reset_frequency"],
        shrink_factor=p["shrink_factor"],
        perturb_factor=p["perturb_factor"],
        min_value=p["min_value"],
        max_value=p["max_value"],
        spr_weight=p["spr_weight"],
        adam_eps=1.5e-4,
        adam_weight_decay=0.1,
    )
    train(train_key, p, agent, env, rb)


if __name__ == "__main__":
    run()
