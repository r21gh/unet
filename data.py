import glob
import os
import tensorflow as tf
import numpy as np

PATH = 'datasets/'
BATCH_SIZE = 12
IMAGE_SIZE = 512


def read_train_data():
    x_files = [f for f in glob.glob(PATH + "train/*.png", recursive=True)]
    y_files = [f for f in glob.glob(PATH + "mask/*.png", recursive=True)]

    def read_image(x_filename, y_filename):
        x_image_string = tf.io.read_file(x_filename)
        y_image_string = tf.io.read_file(y_filename)

        x_image_decoded = tf.image.decode_png(x_image_string, channels=3)
        y_image_decoded = tf.image.decode_png(y_image_string, channels=1)

        x_image_resized = tf.image.resize(x_image_decoded, [IMAGE_SIZE, IMAGE_SIZE])
        y_image_resized = tf.image.resize(y_image_decoded, [IMAGE_SIZE, IMAGE_SIZE])

        x_image_norm = x_image_resized / 255
        y_image_norm = y_image_resized / 255

        return x_image_norm, y_image_norm

    dataset = tf.data.Dataset.from_tensor_slices((x_files, y_files))

    dataset = dataset.map(read_image).shuffle(1000).batch(BATCH_SIZE)

    return dataset


def read_predict_data():
    file_path = 'predict/pre.jpg'
    image = tf.io.read_file(file_path)
    image_decoded = tf.image.decode_jpeg(image, channels=3)
    image_resized = tf.image.resize(image_decoded, [IMAGE_SIZE, IMAGE_SIZE])
    image_norm = image_resized / 255

    return image_norm


def save_image(image):
    file_path = 'predict/result.jpg'
    image = image * 255

    encode_image = tf.image.encode_jpeg(image, format='rgb', quality=100)

    with open(file_path, 'wb') as fd:
        fd.write(encode_image)