from functools import partial
import jax

from jax import numpy as jnp
from flax.training.train_state import TrainState
from flax.training import checkpoints

from data_loader import read_train_data

CKPT_DIR = 'checkpoints'
IMAGE_SIZE = 512


class CustomTrainState(TrainState):
    batch_stats: dict

    def apply_fn_with_bn(self, *args, **kwargs):
        output, mutated_vars = self.apply_fn(*args, **kwargs,
                                             mutable=["batch_stats"], rngs={'dropout': jax.random.PRNGKey(2)})
        new_batch_stats = mutated_vars["batch_stats"]
        return output, new_batch_stats

    def update_batch_stats(self, new_batch_stats):
        return self.replace(batch_stats=new_batch_stats)
    
@jax.jit
def dice_coef(y_true, y_pred):
    y_true = jnp.ravel(y_true)
    y_pred = jnp.ravel(y_pred)
    intersection = jnp.sum(y_true * y_pred)
    return 2.0 * intersection / (jnp.sum(y_true) + jnp.sum(y_pred) + 1)


@jax.jit
def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

@partial(jax.jit, static_argnums=(3,))
def train_step(x, y, train_state, is_training=True):
    def loss_fn(params, batch_stats, is_training):
        y_pred, batch_stats = train_state.apply_fn_with_bn({"params": params, "batch_stats": batch_stats}, x)
        loss = dice_coef_loss(y, y_pred)

        return loss, batch_stats

    if is_training:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, batch_stats), grads = grad_fn(train_state.params, train_state.batch_stats, True)

        train_state = train_state.apply_gradients(grads=grads)
        train_state = train_state.update_batch_stats(batch_stats)
    else:
        loss, batch_stats = loss_fn(train_state.params, train_state.batch_stats, False)

    return loss, train_state