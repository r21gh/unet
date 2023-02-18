import time
from functools import partial
import jax
import optax

from jax import numpy as jnp
from flax.training.train_state import TrainState
from flax.training import checkpoints

from model import UNet
from training import CustomTrainState, train_step
from data_loader import read_predict_data, save_image

CKPT_DIR = 'checkpoints'
IMAGE_SIZE = 512

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