from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import SPRNet
from slimdqn.sample_collection.subseq_replay_buffer import SubsequenceReplayBuffer, SubsequenceReplayElement


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
        horizon_cycle_steps: int,
        update_to_data: int,
        target_update_frequency: int,
        target_update_tau: float,
        reset_frequency: int,
        shrink_factor: float,
        perturb_factor: float,
        min_value: float,
        max_value: float,
        spr_weight: float,
        adam_eps: float = 1e-8,
    ):
        self.n_bins = n_bins
        self.network = SPRNet(features, n_actions, n_bins)

        self.network.apply_fn = lambda params, states, actions: self.network.apply(params, states, actions)
        self.apply_fn_inference = lambda params, state: self.network.apply(params, state)

        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.min_update_horizon = min_update_horizon
        self.max_update_horizon = max_update_horizon
        self.horizon_cycle_steps = horizon_cycle_steps
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.target_update_tau = target_update_tau
        self.reset_frequency = reset_frequency
        self.shrink_factor = shrink_factor
        self.perturb_factor = perturb_factor
        self.spr_weight = spr_weight
        self.cumulated_loss = 0
        self.cumulated_unsupported_prob = 0
        self.support = jnp.linspace(min_value, max_value, self.n_bins, dtype=jnp.float32)

    def update_online_params(self, step: int, replay_buffer: SubsequenceReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples, metadata = replay_buffer.sample()

            self.params, self.optimizer_state, losses, unsupported_prob = self.learn_on_batch(
                self.params, self.optimizer_state, batch_samples, metadata["probabilities"]
            )
            metadata.update({"loss": losses})
            replay_buffer.update(metadata)

            self.cumulated_loss += losses.mean()
            self.cumulated_unsupported_prob += unsupported_prob

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:

            logs = {
                "loss": self.cumulated_loss / (self.target_update_frequency / self.update_to_data),
                "unsupported_prob": self.cumulated_unsupported_prob
                / (self.target_update_frequency / self.update_to_data),
            }
            self.cumulated_loss = 0
            self.cumulated_unsupported_prob = 0

            return True, logs
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, optimizer_state, batch_samples, batch_probabilities):
        grad_loss, (losses, unsupported_prob) = jax.grad(self.loss_on_batch, has_aux=True)(
            params, batch_samples, batch_probabilities
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, losses, unsupported_prob

    def loss_on_batch(self, params: FrozenDict, samples, batch_probabilities):
        losses, unsupported_probs = jax.vmap(self.loss, in_axes=(None, None, 0))(params, samples)
        loss_weights = 1.0 / jnp.sqrt(batch_probabilities + 1e-10)  # sqrt because beta is fixed to 0.5 in PER
        loss_weights /= jnp.max(loss_weights)
        return (losses * loss_weights).mean(), (losses, unsupported_probs.mean())

    def loss(self, params: FrozenDict, sample: SubsequenceReplayElement):
        # computes the loss for a single sample
        target_support, target_prob = self.compute_target(params, sample)
        q_logits, latent_predictions, latent_targets = self.network.apply_fn(
            params, sample.states_stack, sample.actions_stack
        )
        q_logits = q_logits[sample.action[0]]
        projected_target, unsupported_prob = self.project_target_on_support(target_support, target_prob)
        cross_entropy = optax.softmax_cross_entropy(q_logits, jax.lax.stop_gradient(projected_target))
        spr_loss = optax.squared_error(latent_predictions, jax.lax.stop_gradient(latent_targets))
        spr_loss = spr_loss * sample.sample_trajectory_mask
        return cross_entropy + self.spr_weight * spr_loss, unsupported_prob

    def compute_target(self, params: FrozenDict, sample: SubsequenceReplayElement):
        # computes the target value for single sample
        target_support = sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * self.support
        target_logits = self.apply_fn_inference(params, sample.next_state)
        target_prob = jax.nn.softmax(target_logits[jnp.argmax(jax.nn.softmax(target_logits) @ self.support)])
        return target_support, target_prob

    def project_target_on_support(self, target_support: jax.Array, target_prob: jax.Array) -> jax.Array:
        delta_z = (self.support[-1] - self.support[0]) / (self.n_bins - 1)
        clipped_support = jnp.clip(target_support, self.support[0], self.support[-1])
        return (
            jnp.clip(1 - jnp.abs(clipped_support - self.support[:, None]) / delta_z, 0, 1) @ target_prob,
            (
                (
                    jnp.clip(1 - jnp.abs(clipped_support - self.support[:, None]) / delta_z, 0, 1)[jnp.array([0, -1])]
                    == 1  # just take probabilities beyond vmin (0) and vmax (-1)
                )
                @ target_prob
            ).sum(),
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, **kwargs):
        # computes the best action for a single state
        # We first compute the probabilities by applying the softmax on the last axis (bin axis).
        # Then, we compute the expectation by multiplying with the bin centers.
        return jnp.argmax(jax.nn.softmax(self.network.apply_fn(params, state)) @ self.support)

    def get_model(self):
        return {"params": self.params}
