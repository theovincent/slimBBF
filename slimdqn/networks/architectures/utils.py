import collections
import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


@functools.partial(jax.vmap, in_axes=(0, 0, 0, None))
def _crop_with_indices(img, x, y, cropped_shape):
    cropped_image = jax.lax.dynamic_slice(img, [x, y, 0], cropped_shape[1:])
    return cropped_image


def _per_image_random_crop(key, img, cropped_shape):
    """Random crop an image."""
    batch_size, width, height = cropped_shape[:-1]
    key_x, key_y = jax.random.split(key, 2)
    x = jax.random.randint(key_x, shape=(batch_size,), minval=0, maxval=img.shape[1] - width)
    y = jax.random.randint(key_y, shape=(batch_size,), minval=0, maxval=img.shape[2] - height)
    return _crop_with_indices(img, x, y, cropped_shape)


def _intensity_aug(key, x, scale=0.05):
    """Follows the code in Schwarzer et al. (2020) for intensity augmentation."""
    r = jax.random.normal(key, shape=(x.shape[0], 1, 1, 1))
    noise = 1.0 + (scale * jnp.clip(r, -2.0, 2.0))
    return x * noise


@functools.partial(jax.jit)
def drq_augment(key, obs, img_pad=4):
    """Padding and cropping for DrQ."""
    flat_obs = obs.reshape(-1, *obs.shape[-3:])
    paddings = [(0, 0), (img_pad, img_pad), (img_pad, img_pad), (0, 0)]
    cropped_shape = flat_obs.shape
    # The reference uses ReplicationPad2d in pytorch, but it is not available in Jax. Use 'edge' instead.
    flat_obs = jnp.pad(flat_obs, paddings, "edge")
    key1, key2 = jax.random.split(key, num=2)
    cropped_obs = _per_image_random_crop(key2, flat_obs, cropped_shape)
    aug_obs = _intensity_aug(key1, cropped_obs)
    return aug_obs.reshape(*obs.shape)


def normalize_and_augment(x, rng):
    """Input normalization and if specified, data augmentation."""
    out = x.astype(jnp.float32) / 255.0
    out = drq_augment(rng, out)
    return out


def renormalize(tensor, has_batch=False):
    shape = tensor.shape
    if not has_batch:
        tensor = jnp.expand_dims(tensor, 0)
    tensor = tensor.reshape(tensor.shape[0], -1)
    max_value = jnp.max(tensor, axis=-1, keepdims=True)
    min_value = jnp.min(tensor, axis=-1, keepdims=True)
    return ((tensor - min_value) / (max_value - min_value + 1e-5)).reshape(*shape)


def copy_params(source, target, keys):
    """Copies a set of keys from source to target."""
    if isinstance(source, dict) or isinstance(source, collections.OrderedDict) or isinstance(source, FrozenDict):
        fresh_dict = {}
        for k, v in source.items():
            if k in keys:
                fresh_dict[k] = v
            else:
                fresh_dict[k] = copy_params(source[k], target[k], keys)
        return fresh_dict
    else:
        return target


@functools.partial(jax.jit, static_argnames=("keys"))
def interpolate_weights(old_params, new_params, old_weight, new_weight, keys):

    old_params = old_params["params"]
    new_params = new_params["params"]

    def combination(old_param, new_param):
        return old_param * old_weight + new_param * new_weight

    combined_params = {}
    if keys is None:
        keys = old_params.keys()
    for k in keys:
        combined_params[k] = jax.tree_util.tree_map(combination, old_params[k], new_params[k])
    for k, v in old_params.items():
        if k not in keys:
            combined_params[k] = v

    return {"params": combined_params}


def exponential_decay_scheduler(decay_period, warmup_steps, initial_value, final_value, reverse=False):
    """Instantiate a logarithmic schedule for a parameter.

    By default the extreme point to or from which values decay logarithmically
    is 0, while changes near 1 are fast. In cases where this may not
    be correct (e.g., lambda) pass reversed=True to get proper
    exponential scaling.

    Args:
        decay_period: float, the period over which the value is decayed.
        warmup_steps: int, the number of steps taken before decay starts.
        initial_value: float, the starting value for the parameter.
        final_value: float, the final value for the parameter.
        reverse: bool, whether to treat 1 as the asmpytote instead of 0.

    Returns:
        A decay function mapping step to parameter value.
    """
    if reverse:
        initial_value = 1 - initial_value
        final_value = 1 - final_value

    start = np.log(initial_value)
    end = np.log(final_value)

    if decay_period == 0:
        return lambda x: initial_value if x < warmup_steps else final_value

    def scheduler(step):
        steps_left = decay_period + warmup_steps - step
        bonus_frac = steps_left / decay_period
        bonus = np.clip(bonus_frac, 0.0, 1.0)
        new_value = bonus * (start - end) + end

        new_value = np.exp(new_value)
        if reverse:
            new_value = 1 - new_value
        return new_value

    return scheduler
