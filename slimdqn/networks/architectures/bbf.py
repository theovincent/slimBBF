from typing import Sequence

import jax
import flax.linen as nn
import jax.numpy as jnp


class Stack(nn.Module):
    stack_size: int

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        x = nn.Conv(features=self.stack_size, kernel_size=(3, 3), kernel_init=initializer)(x)
        x = nn.max_pool(x, window_shape=(3, 3), padding="SAME", strides=(2, 2))

        for _ in range(2):
            block_input = x
            x = nn.relu(x)
            x = nn.Conv(features=self.stack_size, kernel_size=(3, 3), kernel_init=initializer)(x)
            x = nn.relu(x)
            x = nn.Conv(features=self.stack_size, kernel_size=(3, 3), kernel_init=initializer)(x)
            x += block_input

        return x


class ImpalaEncoder(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feature in self.features:
            x = Stack(feature)(x)
        return nn.relu(x)


def max_min_normalize(x):
    x_min = x.min()
    return (x - x_min) / (x.max() - x_min + 1e-5)


class TransitionModel(nn.Module):
    n_actions: int
    latent_dim: int

    @nn.compact
    def __call__(self, x, action):
        initializer = nn.initializers.xavier_uniform()
        # x shape (height, width, n_channels)
        # shape (n_actions)
        action_onehot = jax.nn.one_hot(action, self.n_actions)
        # shape (height, width, n_actions)
        action_channels = jax.lax.broadcast(action_onehot, (x.shape[0], x.shape[1]))
        # shape (height, width, n_channels + n_actions)
        x = jnp.concatenate([x, action_channels], -1)
        x = nn.Conv(features=self.latent_dim, kernel_size=(3, 3), kernel_init=initializer)(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.latent_dim, kernel_size=(3, 3), kernel_init=initializer)(x)
        x = nn.relu(x)
        x = max_min_normalize(x)
        return x, x


class MultiStepTransitionModel(nn.Module):
    n_actions: int
    latent_dim: int

    @nn.compact
    def __call__(self, latent, actions):
        scan = nn.scan(TransitionModel, variable_broadcast=["params"])(self.n_actions, self.latent_dim)
        # only return the sequence of latents
        return scan(latent, actions)[1]


class SPRNet(nn.Module):
    features: Sequence[int]
    n_actions: int
    n_bins: int

    def setup(self):
        self.encoder = ImpalaEncoder(self.features[:3])
        self.transition_model = MultiStepTransitionModel(self.n_actions, self.features[2])
        self.projector = nn.Dense(self.features[3], kernel_init=nn.initializers.xavier_uniform())
        self.predictor = nn.Dense(self.features[3], kernel_init=nn.initializers.xavier_uniform())
        self.q_logits_head = nn.Dense(self.n_actions * self.n_bins, kernel_init=nn.initializers.xavier_uniform())

    def spr_rollout(self, latent, actions):
        pred_latents = self.transition_model(latent, actions)
        representations = pred_latents.reshape(pred_latents.shape[0], -1)
        projected_representations = jax.vmap(self.projector)(representations)
        predictions = jax.vmap(self.predictor)(projected_representations)
        return predictions

    def encode_project(self, x):
        representation = max_min_normalize(self.encoder(x))
        representation = representation.reshape(representation.shape[0], -1)
        return self.projector(representation)

    @nn.compact
    def __call__(self, x, actions=None):
        # Normalize input?
        spatial_latent = max_min_normalize(self.encoder(x))
        representation = spatial_latent.reshape(-1)
        x = nn.relu(self.projector(representation))

        x = x[0] if x.ndim > 1 else x
        q_logits = self.q_logits_head(x).reshape((self.n_actions, self.n_bins))

        if actions is None:
            return q_logits

        spr_predictions = self.spr_rollout(spatial_latent, actions)
        return q_logits, spr_predictions
