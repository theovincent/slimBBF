from typing import Sequence

import jax
import flax.linen as nn
import jax.numpy as jnp

from slimdqn.networks.architectures.utils import normalize_and_augment, renormalize


class Stack(nn.Module):
    """Stack of pooling and convolutional blocks with residual connections."""

    stack_size: int

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        x = nn.Conv(
            features=self.stack_size,
            kernel_size=(3, 3),
            kernel_init=initializer,
        )(x)
        x = nn.max_pool(x, window_shape=(3, 3), padding="SAME", strides=(2, 2))

        for _ in range(2):
            block_input = x
            x = nn.relu(x)
            x = nn.relu(nn.Conv(features=self.stack_size, kernel_size=(3, 3))(x))
            x = nn.Conv(features=self.stack_size, kernel_size=(3, 3))(x)
            x += block_input

        return x


class ImpalaEncoder(nn.Module):

    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = Stack(self.features[0])(x)
        x = Stack(self.features[1])(x)
        x = nn.relu(Stack(self.features[2])(x))
        return x


class DQNNet(nn.Module):
    features: Sequence[int]
    architecture_type: str
    n_actions: int

    @nn.compact
    def __call__(self, x):
        if self.architecture_type == "cnn":
            initializer = nn.initializers.xavier_uniform()
            idx_feature_start = 3
            x = nn.relu(
                nn.Conv(features=self.features[0], kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer)(
                    jnp.array(x, ndmin=4) / 255.0
                )
            )
            x = nn.relu(
                nn.Conv(features=self.features[1], kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer)(x)
            )
            x = nn.relu(
                nn.Conv(features=self.features[2], kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer)(x)
            )
            x = x.reshape((x.shape[0], -1))
        elif self.architecture_type == "der":
            initializer = nn.initializers.xavier_uniform()
            idx_feature_start = 2
            x = ImpalaEncoder(self.features[:idx_feature_start])(jnp.array(x, ndmin=4) / 255.0)
            x = x.reshape((x.shape[0], -1))
        elif self.architecture_type == "impala":
            initializer = nn.initializers.xavier_uniform()
            idx_feature_start = 3
            x = ImpalaEncoder(self.features[:idx_feature_start])(jnp.array(x, ndmin=4) / 255.0)
            x = x.reshape((x.shape[0], -1))
        elif self.architecture_type == "fc":
            initializer = nn.initializers.lecun_normal()
            idx_feature_start = 0

        x = jnp.squeeze(x)

        for idx_layer in range(idx_feature_start, len(self.features)):
            x = nn.relu((nn.Dense(self.features[idx_layer], kernel_init=initializer)(x)))

        return nn.Dense(self.n_actions, kernel_init=initializer)(x)


class ConvTMCell(nn.Module):
    """MuZero-style transition model for SPR."""

    n_actions: int
    latent_dim: int

    @nn.compact
    def __call__(self, x, action):
        action_onehot = jax.nn.one_hot(action, self.n_actions)
        action_onehot = jax.lax.broadcast(action_onehot, (x.shape[-3], x.shape[-2]))
        x = jnp.concatenate([x, action_onehot], -1)
        x = nn.Conv(
            features=self.latent_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=nn.initializers.xavier_uniform(),
            dtype=jnp.float32,
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=self.latent_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=nn.initializers.xavier_uniform(),
            dtype=jnp.float32,
        )(x)
        x = nn.relu(x)

        x = renormalize(x)

        return x, x


class TransitionModel(nn.Module):
    """An SPR-style transition model (uses ConvTMCell)."""

    n_actions: int
    latent_dim: int

    @nn.compact
    def __call__(self, x, action):
        scan = nn.scan(
            ConvTMCell,
            in_axes=0,
            out_axes=0,
            variable_broadcast=["params"],
            split_rngs={"params": False},
        )(latent_dim=self.latent_dim, n_actions=self.n_actions)
        return scan(x, action)


class SPRNet(nn.Module):
    features: Sequence[int]
    n_actions: int
    n_bins: int

    def setup(self):
        self.encoder = ImpalaEncoder(features=self.features[:3])
        self.transition_model = TransitionModel(n_actions=self.n_actions, latent_dim=self.features[2])
        self.projection = nn.Dense(self.features[3], kernel_init=nn.initializers.xavier_uniform())
        self.predictor = nn.Dense(self.features[3], kernel_init=nn.initializers.xavier_uniform())
        self.q_logits_head = nn.Dense(self.n_actions * self.n_bins, kernel_init=nn.initializers.xavier_uniform())

    def spr_rollout(self, latent, actions):
        _, pred_latents = self.transition_model(latent, actions)
        representations = pred_latents.reshape(pred_latents.shape[0], -1)
        projected_representations = jax.vmap(self.projection)(representations)
        predictions = jax.vmap(self.predictor)(projected_representations)
        return predictions

    def encode_project(self, x):
        representation = renormalize(self.encoder(x))
        representation = representation.reshape(representation.shape[0], -1)
        return self.projection(representation)

    @nn.compact
    def __call__(self, x, actions=None, rng=None, do_rollout=True):
        x = normalize_and_augment(x, rng)
        spatial_latent = renormalize(self.encoder(x))
        if actions is None or not do_rollout:
            representation = spatial_latent.reshape(-1)
        else:
            representation = spatial_latent.reshape(spatial_latent.shape[0], -1)

        # Single hidden layer
        x = self.projection(representation)
        x = nn.relu(x)

        if x.ndim > 1:
            assert actions is not None
            x = x[0]
        q_logits = self.q_logits_head(x).reshape((self.n_actions, self.n_bins))

        if actions is None:
            return q_logits

        spr_predictions = self.spr_rollout(spatial_latent, actions)
        return q_logits, spr_predictions
