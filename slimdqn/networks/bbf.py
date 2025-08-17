from functools import partial
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import SPRNet
from slimdqn.networks.architectures.utils import (
    copy_params,
    interpolate_weights,
    exponential_scheduler,
    normalize_and_augment,
)
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
        gamma: float,
        update_horizon: int,
        max_update_horizon: int,
        horizon_cycle_steps: int,
        update_to_data: int,
        target_update_tau: float,
        reset_frequency: int,
        shrink_factor: float,
        perturb_factor: float,
        min_value: float,
        max_value: float,
        spr_weight: float,
        adam_eps: float = 1e-8,
        adam_weight_decay: float = 0.1,
    ):
        self.n_bins = n_bins
        self.observation_dim = observation_dim
        self.network = SPRNet(features, n_actions, n_bins)

        self.key = key

        param_key, init_key, self.key = jax.random.split(self.key, 3)
        self.params = self.network.init(
            x=jnp.zeros(observation_dim, dtype=jnp.float32),
            actions=jnp.zeros((5,)),
            rngs={"params": param_key},
        )

        optimizer = optax.adamw(
            learning_rate,
            eps=adam_eps,
            weight_decay=adam_weight_decay,
            mask=lambda p: jax.tree_util.tree_map(lambda x: x.ndim != 1, p),  # bias not decayed
        )

        self.optimizer = optax.chain(
            optax.masked(
                optimizer, {"params": {k: k in ("encoder", "transition_model") for k in self.params["params"]}}
            ),
            optax.masked(
                optimizer, {"params": {k: k not in ("encoder", "transition_model") for k in self.params["params"]}}
            ),
        )
        self.optimizer_state = self.optimizer.init(self.params)
        self.target_params = self.params

        self.min_gamma = min_gamma
        self.gamma = gamma
        self.update_horizon = update_horizon
        self.max_update_horizon = max_update_horizon
        self.gamma_scheduler = exponential_scheduler(horizon_cycle_steps, min_gamma, gamma)
        self.update_horizon_scheduler = lambda t: int(
            exponential_scheduler(horizon_cycle_steps, update_horizon, max_update_horizon)(t)
        )
        self.horizon_cycle_grad_steps = 0  # to track number of grad steps for schedulers
        self.update_to_data = update_to_data
        self.target_update_tau = target_update_tau
        self.reset_frequency = reset_frequency
        self.shrink_factor = shrink_factor
        self.perturb_factor = perturb_factor
        self.spr_weight = spr_weight
        self.cumulated_loss = 0
        self.support = jnp.linspace(min_value, max_value, n_bins, dtype=jnp.float32)

    @partial(jax.jit, static_argnames="self")
    def apply_multiple_updates(self, params, params_target, optimizer_state, batches, probabilities):
        def apply_single_update(state, batches_and_probability):
            batch, batch_probability = batches_and_probability
            params, optimizer_state, loss = self.learn_on_batch(
                state[0], params_target, state[1], batch, batch_probability
            )
            return (params, optimizer_state), (loss)

        batches = jax.tree.map(lambda *batch: jnp.stack(batch), *batches)
        probabilities = jax.tree.map(lambda *probability: jnp.stack(probability), *probabilities)

        (final_params, final_optimizer_state), losses = jax.lax.scan(
            apply_single_update, (params, optimizer_state), (batches, probabilities)
        )
        return final_params, final_optimizer_state, losses

    def update_online_params(self, step: int, replay_buffer: SubsequenceReplayBuffer):
        batches_and_metadatas = [
            replay_buffer.sample(
                n=self.update_horizon_scheduler(self.horizon_cycle_grad_steps),
                gamma=self.gamma_scheduler(self.horizon_cycle_grad_steps),
            )
            for _ in range(self.update_to_data)
        ]
        batches = list(map(lambda x: x[0], batches_and_metadatas))
        indices = jnp.array(list(map(lambda x: x[1]["indices"], batches_and_metadatas)))
        probabilities = jnp.array(list(map(lambda x: x[1]["probabilities"], batches_and_metadatas)))
        self.params, self.optimizer_state, per_sample_loss = self.apply_multiple_updates(
            self.params, self.target_params, self.optimizer_state, batches, probabilities
        )
        replay_buffer.update({"loss": per_sample_loss.reshape(-1), "indices": indices.reshape(-1)})
        self.cumulated_loss += jnp.mean(per_sample_loss)
        self.horizon_cycle_grad_steps += self.update_to_data

    def update_target_params(self, step):
        self.target_params = interpolate_weights(
            self.target_params,
            self.params,
            old_weight=1 - self.target_update_tau,
            new_weight=self.target_update_tau,
            keys=None,  # all keys
        )

        logs = {"loss": self.cumulated_loss / self.update_to_data}
        self.cumulated_loss = 0

        return True, logs  # return True to keep consistent with dqn code

    def reset_network_params(self, step: int):
        if step % self.reset_frequency == 0:
            reset_key, self.key = jax.random.split(self.key)
            self.params, self.target_params, self.optimizer_state = self.apply_reset_params(
                self.params, self.target_params, self.optimizer_state, reset_key
            )
            self.horizon_cycle_grad_steps = 0

    @partial(jax.jit, static_argnames="self")
    def apply_reset_params(self, params, target_params, optimizer_state, reset_key):
        online_key, target_key = jax.random.split(reset_key, 2)
        random_params = self.network.init(
            x=jnp.zeros(self.observation_dim, dtype=jnp.float32),
            actions=jnp.zeros((5,)),
            rngs={"params": online_key},
        )
        target_random_params = self.network.init(
            x=jnp.zeros(self.observation_dim, dtype=jnp.float32),
            actions=jnp.zeros((5,)),
            rngs={"params": target_key},
        )

        params = interpolate_weights(
            old_params=params,
            new_params=random_params,
            old_weight=self.shrink_factor,
            new_weight=self.perturb_factor,
            keys=("encoder", "transition_model"),  # for conv layers, shrink and perturb
        )
        params = copy_params(
            params, random_params, keys=("encoder", "transition_model")
        )  # for other layers, full reset

        updated_optim_state = []
        optim_state = self.optimizer.init(params)
        for i in range(len(optim_state)):
            optim_to_copy = copy_params(
                dict(optimizer_state[i]._asdict()), dict(optim_state[i]._asdict()), keys=("encoder", "transition_model")
            )
            updated_optim_state.append(optim_state[i]._replace(**optim_to_copy))
        optimizer_state = tuple(updated_optim_state)

        target_params = interpolate_weights(
            old_params=target_params,
            new_params=target_random_params,
            old_weight=self.shrink_factor,
            new_weight=self.perturb_factor,
            keys=("encoder", "transition_model"),
        )
        target_params = copy_params(target_params, target_random_params, keys=("encoder", "transition_model"))

        return params, target_params, optimizer_state

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: FrozenDict, params_target: FrozenDict, optimizer_state, batch_samples, batch_probabilities
    ):
        grad_loss, (losses) = jax.grad(self.loss_on_batch, has_aux=True)(
            params, params_target, batch_samples, batch_probabilities
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, losses

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples, batch_probabilities):
        augment_s_key, augment_ns_key, self.key = jax.random.split(self.key, 3)
        samples = replace(samples, states_stack=normalize_and_augment(samples.states_stack, augment_s_key))
        samples = replace(samples, next_state=normalize_and_augment(samples.next_state, augment_ns_key))
        losses, td_losses = jax.vmap(self.loss, in_axes=(None, None, 0))(params, params_target, samples)
        loss_weights = 1.0 / jnp.sqrt(batch_probabilities + 1e-10)  # sqrt because beta is fixed to 0.5 in PER
        loss_weights /= jnp.max(loss_weights)
        return (losses * loss_weights).mean(), (td_losses)

    def loss(self, params: FrozenDict, params_target: FrozenDict, sample: SubsequenceReplayElement):
        # computes the loss for a single sample
        spr_targets = self.network.apply(params_target, sample.states_stack[1:], method=self.network.encode_project)
        target_support, target_prob = self.compute_target(params_target, sample)
        projected_target = self.project_target_on_support(target_support, target_prob)
        q_logits, spr_predictions = self.network.apply(params, sample.states_stack[0], sample.actions_stack[:-1])
        q_logits = q_logits[sample.actions_stack[0]]

        cross_entropy = optax.softmax_cross_entropy(q_logits, jax.lax.stop_gradient(projected_target))

        spr_predictions = spr_predictions / jnp.linalg.norm(spr_predictions, 2, -1, keepdims=True)
        spr_targets = spr_targets / jnp.linalg.norm(spr_targets, 2, -1, keepdims=True)

        spr_loss = jnp.power(spr_predictions - jax.lax.stop_gradient(spr_targets), 2).sum(-1)
        spr_loss = (spr_loss * sample.same_trajectory_mask[:-1]).mean(0)
        return cross_entropy + self.spr_weight * spr_loss, (cross_entropy)

    def compute_target(self, params: FrozenDict, sample: SubsequenceReplayElement):
        # computes the target value for single sample
        target_support = sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * self.support
        target_logits = self.network.apply(params, sample.next_state)
        target_prob = jax.nn.softmax(target_logits[jnp.argmax(jax.nn.softmax(target_logits) @ self.support)])
        return target_support, target_prob

    def project_target_on_support(self, target_support: jax.Array, target_prob: jax.Array) -> jax.Array:
        delta_z = (self.support[-1] - self.support[0]) / (self.n_bins - 1)
        clipped_support = jnp.clip(target_support, self.support[0], self.support[-1])
        return jnp.clip(1 - jnp.abs(clipped_support - self.support[:, None]) / delta_z, 0, 1) @ target_prob

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, **kwargs):
        # computes the best action for a single state
        # We first compute the probabilities by applying the softmax on the last axis (bin axis).
        # Then, we compute the expectation by multiplying with the bin centers.
        return jnp.argmax(jax.nn.softmax(self.network.apply(params, state)) @ self.support)

    def get_model(self):
        return {"params": self.params}
