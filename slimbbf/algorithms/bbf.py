from functools import partial
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimbbf.algorithms.architectures.bbf import BBFNet
from slimbbf.algorithms.architectures.utils import (
    exponential_scheduler,
    reverse_exponential_scheduler,
    normalize_and_augment,
)
from slimbbf.sample_collection.subseq_replay_buffer import SubsequenceReplayBuffer, SubsequenceReplayElement


class BBF:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        n_bins: int,
        features: list,
        learning_rate: float,
        min_gamma: float,
        max_gamma: float,
        min_update_horizon: int,
        max_update_horizon: int,
        gamma_horizon_decay_steps: int,
        update_to_data: int,
        tau: float,
        spr_steps: int,
    ):
        self.observation_dim = observation_dim
        self.spr_steps = spr_steps
        self.key, init_key = jax.random.split(key)

        self.network = BBFNet(features, n_actions, n_bins)
        self.params = self.network.init(
            init_key, jnp.zeros(observation_dim, dtype=jnp.float32), jnp.zeros(spr_steps, dtype=int)
        )
        self.bins = np.linspace(start=-10, stop=10, num=n_bins)

        self.optimizer = optax.adamw(
            learning_rate,
            eps=1.5e-4,
            weight_decay=0.1,
            mask=lambda p: jax.tree_util.tree_map(lambda x: x.ndim != 1, p),  # bias not decayed
        )
        self.optimizer_state = self.optimizer.init(self.params)
        self.target_params = self.params.copy()

        self.update_horizon_schedule = exponential_scheduler(
            gamma_horizon_decay_steps, max_update_horizon, min_update_horizon
        )
        self.gamma_schedule = reverse_exponential_scheduler(gamma_horizon_decay_steps, min_gamma, max_gamma)
        self.update_to_data = update_to_data
        self.tau = tau
        self.grad_steps_after_reset = 0  # to track number of grad steps for schedulers
        self.cumulated_td_loss = 0
        self.cumulated_spr_loss = 0

    def update_online_params(self, replay_buffer: SubsequenceReplayBuffer):
        # Compute effective grad step to use same n and gamma for update_to_data updates
        effective_grad_step_after_reset = self.grad_steps_after_reset // self.update_to_data * self.update_to_data
        update_horizon = int(np.round(self.update_horizon_schedule(effective_grad_step_after_reset)))
        gamma = self.gamma_schedule(effective_grad_step_after_reset)
        samples, indices, importance_weights = replay_buffer.sample(n=update_horizon, gamma=gamma)
        self.key, key = jax.random.split(self.key)

        self.params, self.target_params, self.optimizer_state, per_sample_td_loss, spr_loss = self.learn_on_batch(
            self.params,
            self.target_params,
            self.optimizer_state,
            samples,
            importance_weights,
            gamma**update_horizon,
            key,
        )

        replay_buffer.update(indices, per_sample_td_loss)
        self.cumulated_td_loss = (1 - self.tau) * self.cumulated_td_loss + self.tau * per_sample_td_loss.mean()
        self.cumulated_spr_loss = (1 - self.tau) * self.cumulated_spr_loss + self.tau * spr_loss
        self.grad_steps_after_reset += 1

    def reset_params(self):
        self.key, key = jax.random.split(self.key)
        self.params, self.target_params, self.optimizer_state = self.apply_reset_params(
            self.params, self.target_params, self.optimizer_state, key
        )
        self.grad_steps_after_reset = 0

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        params: FrozenDict,
        params_target: FrozenDict,
        optimizer_state,
        samples,
        importance_weights,
        discounted_gamma,
        key,
    ):
        grad_loss, (per_sample_td_loss, spr_loss) = jax.grad(self.loss_on_batch, has_aux=True)(
            params, params_target, samples, importance_weights, discounted_gamma, key
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        params_target = optax.incremental_update(params, params_target, self.tau)

        return params, params_target, optimizer_state, per_sample_td_loss, spr_loss

    def loss_on_batch(
        self, params: FrozenDict, params_target: FrozenDict, samples, importance_weights, discounted_gamma, key
    ):
        states_stack_key, next_state_key = jax.random.split(key)
        samples = replace(samples, states_stack=normalize_and_augment(samples.states_stack, states_stack_key))
        samples = replace(samples, next_state=normalize_and_augment(samples.next_state, next_state_key))

        losses, td_losses, spr_losses = jax.vmap(self.loss, in_axes=(None, None, 0, 0, None))(
            params, params_target, samples, importance_weights, discounted_gamma
        )

        return losses.mean(), (td_losses, spr_losses.mean())

    def loss(
        self,
        params: FrozenDict,
        params_target: FrozenDict,
        sample: SubsequenceReplayElement,
        importance_weight: float,
        discounted_gamma: float,
    ):
        # Only works for a single sample
        target_probs = self.compute_target(params, params_target, sample, discounted_gamma)
        q_logits, spr_predictions = self.network.apply(params, sample.states_stack[0], sample.actions_stack[:-1])
        cross_entropy = importance_weight * optax.softmax_cross_entropy(q_logits, jax.lax.stop_gradient(target_probs))

        # shape (window_size, latent_dimension)
        spr_targets = jax.vmap(partial(self.network.apply, method=self.network.encode_and_project), in_axes=(None, 0))(
            params_target, sample.states_stack[1:]
        )
        spr_targets = spr_targets / jnp.linalg.norm(spr_targets, axis=-1, keepdims=True)
        spr_predictions = spr_predictions / jnp.linalg.norm(spr_predictions, axis=-1, keepdims=True)
        # shape (window_size)
        spr_losses = jnp.square(spr_predictions - jax.lax.stop_gradient(spr_targets)).sum(axis=-1)
        # mask out the state that are not from same trajectory
        spr_loss = importance_weight * (spr_losses * sample.same_trajectory_mask[1:]).mean()

        return cross_entropy + 5 * spr_loss, cross_entropy, spr_loss

    def compute_target(
        self, params: FrozenDict, params_target: jax.Array, sample: SubsequenceReplayElement, discounted_gamma: float
    ):
        # computes the target value for single sample using Double DQN update
        # shape (n_actions, n_bins)
        online_probs_actions = jax.nn.softmax(self.network.apply(params, sample.next_state))
        target_probs_actions = jax.nn.softmax(self.network.apply(params_target, sample.next_state))
        target_probs = target_probs_actions[jnp.argmax(online_probs_actions @ self.bins)]

        # shape (n_bins)
        target_locations_ = sample.reward + (1 - sample.is_terminal) * discounted_gamma * self.bins
        targets_locations = jnp.clip(target_locations_, self.bins[0], self.bins[-1])

        def projection(bin_location):
            # Distance to bin location. shape (n_bins)
            distances_to_bin = jnp.abs(targets_locations - bin_location) / (self.bins[1] - self.bins[0])
            # Clip the maximum distance to 1 to only consider the close target locations. shape ()
            return jnp.dot((1 - jnp.minimum(distances_to_bin, 1)), target_probs)

        # shape (n_bins)
        return jax.vmap(projection)(self.bins)

    @partial(jax.jit, static_argnames="self")
    def apply_reset_params(self, params, target_params, optimizer_state, key):
        # Partially reset the encoder and the transition model. Fully reset the rest
        online_key, target_key = jax.random.split(key)
        new_params = self.network.init(
            online_key, jnp.zeros(self.observation_dim, dtype=jnp.float32), jnp.zeros((self.spr_steps,), dtype=int)
        )
        new_target_params = self.network.init(
            target_key, jnp.zeros(self.observation_dim, dtype=jnp.float32), jnp.zeros((self.spr_steps,), dtype=int)
        )

        new_params["params"]["encoder"] = optax.incremental_update(
            new_params["params"]["encoder"], params["params"]["encoder"], 0.5
        )
        new_params["params"]["transition_model"] = optax.incremental_update(
            new_params["params"]["transition_model"], params["params"]["transition_model"], 0.5
        )
        new_target_params["params"]["encoder"] = optax.incremental_update(
            new_target_params["params"]["encoder"], target_params["params"]["encoder"], 0.5
        )
        new_target_params["params"]["transition_model"] = optax.incremental_update(
            new_target_params["params"]["transition_model"], target_params["params"]["transition_model"], 0.5
        )

        new_optimizer_state = self.optimizer.init(new_params)
        new_optimizer_state[0].mu["params"]["encoder"] = optimizer_state[0].mu["params"]["encoder"]
        new_optimizer_state[0].mu["params"]["transition_model"] = optimizer_state[0].mu["params"]["transition_model"]
        new_optimizer_state[0].nu["params"]["encoder"] = optimizer_state[0].nu["params"]["encoder"]
        new_optimizer_state[0].nu["params"]["transition_model"] = optimizer_state[0].nu["params"]["transition_model"]

        return new_params, new_target_params, new_optimizer_state

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        normalized_state = state.astype(jnp.float32) / 255.0
        return jnp.argmax(jax.nn.softmax(self.network.apply(params, normalized_state)) @ self.bins)

    def get_logs(self):
        return {"train/td_loss": self.cumulated_td_loss, "train/spr_loss": self.cumulated_spr_loss}

    def get_model(self):
        return {"params": self.target_params}
