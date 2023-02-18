import time
from functools import partial
import jax
import optax

from jax import numpy as jnp
from flax.training.train_state import TrainState
from flax.training import checkpoints

from model import UNet
from data_loader import read_train_data, read_predict_data, save_image

CKPT_DIR = 'checkpoints'
IMAGE_SIZE = 512


class CustomTrainState(TrainState):
    batch_stats: dict

    def apply_fn_with_bn(self, *args, **kwargs):
        output, mutated_vars = self.apply_fn(*args, **kwargs, mutable=["batch_stats"], rngs={'dropout': jax.random.PRNGKey(2)})
        new_batch_stats = mutated_vars["batch_stats"]
        return output, new_batch_stats

    def update_batch_stats(self, new_batch_stats):
        return self.replace(batch_stats=new_batch_stats)


def dice_coef(y_true, y_pred):
    y_true = jnp.ravel(y_true)
    y_pred = jnp.ravel(y_pred)
    intersection = jnp.sum(y_true * y_pred)
    return 2.0 * intersection / (jnp.sum(y_true) + jnp.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


@partial(jax.jit, static_argnums=(3,))
def train_step(x, y, train_state, is_training=True):
    def loss_fn(params, batch_stats, is_training):
        y_pred, batch_stats = train_state.apply_fn_with_bn({"params": params, "batch_stats": batch_stats}, x,
                                                           is_training=is_training)
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


def main():
    train_set = read_train_data()
    unet = UNet()

    init_rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}

    unet_variables = unet.init(init_rngs, jnp.ones([1, IMAGE_SIZE, IMAGE_SIZE, 3]))

    optimizer = optax.adam(learning_rate=0.001)

    train_state = CustomTrainState.create(apply_fn=unet.apply, params=unet_variables["params"], tx=optimizer,
                                          batch_stats=unet_variables["batch_stats"])

    checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=train_state, step=0, overwrite=True)

    for e in range(20):
        loss_avg = 0
        tic = time.time()
        for x, y in train_set.as_numpy_iterator():
            loss, train_state = train_step(x, y, train_state, True)
            loss_avg += loss

        loss_avg /= len(train_set)
        elapsed = time.time() - tic
        print(f"epoch: {e}, loss: {loss_avg:0.2f}, elapased: {elapsed:0.2f}")


def predict():
    data = read_predict_data()
    unet = UNet(training=False)

    init_rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}

    unet_variables = unet.init(init_rngs, jnp.ones([1, IMAGE_SIZE, IMAGE_SIZE, 3]))

    optimizer = optax.adam(learning_rate=0.001)

    train_state = CustomTrainState.create(apply_fn=unet.apply, params=unet_variables["params"], tx=optimizer,
                                          batch_stats=unet_variables["batch_stats"])

    checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=train_state)

    pred, _ = train_state.apply_fn_with_bn({"params": train_state.params, "batch_stats": train_state.batch_stats}, data)

    save_image(pred)


if __name__ == '__main__':
    main()
    predict()