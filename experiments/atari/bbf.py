import os
import sys

import jax
import numpy as np
import multiprocessing as mp

from experiments.base.bbf import train
from experiments.base.utils import prepare_logs
from slimbbf.environments.atari import AtariEnv
from slimbbf.environments.atari_eval import AtariEval
from slimbbf.algorithms.bbf import BBF
from slimbbf.sample_collection.subsequence_replay_buffer import PrioritizedJaxSubsequenceParallelEnvReplayBuffer


mp.set_start_method("spawn", force=True)


def run(argvs=sys.argv[1:]):
    env_name, algo_name = os.path.abspath(__file__).split("/")[-2], os.path.abspath(__file__).split("/")[-1][:-3]
    p = prepare_logs(env_name, algo_name, argvs)

    q_key, train_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = AtariEnv(p["experiment_name"].split("_")[-1], sticky_actions=False)  # no sticky actions in Atari 100k
    env_eval = lambda n_envs: AtariEval(p["experiment_name"].split("_")[-1], sticky_actions=False, n_envs=n_envs)
    rb = PrioritizedJaxSubsequenceParallelEnvReplayBuffer(
        observation_shape=(env.state_height, env.state_width),
        stack_size=4,
        update_horizon=p["min_update_horizon"],
        gamma=p["max_gamma"],
        subseq_len=6,
        batch_size=p["batch_size"],
        observation_dtype=np.uint8,
    )
    agent = BBF(
        q_key,
        (env.state_height, env.state_width, env.n_stacked_frames),
        env.n_actions,
        n_bins=p["n_bins"],
        features=p["features"],
        learning_rate=p["learning_rate"],
        min_gamma=p["min_gamma"],
        max_gamma=p["max_gamma"],
        min_update_horizon=p["min_update_horizon"],
        max_update_horizon=p["max_update_horizon"],
        gamma_horizon_decay_steps=p["gamma_horizon_decay_steps"],
        update_to_data=p["update_to_data"],
        tau=p["tau"],
        spr_steps=5,
    )

    train(train_key, p, agent, env, env_eval, rb)


if __name__ == "__main__":
    run()
