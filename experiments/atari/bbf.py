import os
import sys

import jax
import numpy as np

from experiments.base.bbf import train, eval
from experiments.base.utils import prepare_logs
from slimbbf.environments.atari import AtariEnv
from slimbbf.environments.atari_eval import AtariEval
from slimbbf.algorithms.bbf import BBF
from slimbbf.sample_collection.subseq_replay_buffer import SubsequenceReplayBuffer


def run(argvs=sys.argv[1:]):
    env_name, algo_name = os.path.abspath(__file__).split("/")[-2], os.path.abspath(__file__).split("/")[-1][:-3]
    p = prepare_logs(env_name, algo_name, argvs)

    q_key, train_key, eval_key = jax.random.split(jax.random.PRNGKey(p["seed"]), 3)

    env = AtariEnv(p["experiment_name"].split("_")[-1], sticky_actions=False)  # no sticky actions in Atari 100k
    rb = SubsequenceReplayBuffer(
        max_capacity=p["replay_buffer_capacity"],
        seed=p["seed"],
        batch_size=p["batch_size"],
        observation_shape=(env.state_height, env.state_width),
        observation_dtype=np.uint8,
        spr_window=5,
        stack_size=4,
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
        max_gamma=p["max_gamma"],
        min_update_horizon=p["min_update_horizon"],
        max_update_horizon=p["max_update_horizon"],
        gamma_horizon_decay_steps=p["gamma_horizon_decay_steps"],
        tau=p["tau"],
        reset_frequency=p["reset_frequency"],
        spr_steps=5,
    )
    train(train_key, p, agent, env, rb)

    if p["eval"]:
        eval_env_key, eval_key = jax.random.split(eval_key)
        env_eval = AtariEval(p["experiment_name"].split("_")[-1], sticky_actions=False, n_envs=100, key=eval_env_key)
        eval(eval_key, p, agent, env_eval)


if __name__ == "__main__":
    run()
