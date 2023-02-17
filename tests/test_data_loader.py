import unittest
import tensorflow as tf
from ..src.data_loader import DataLoader

class TestReadTrainData(unittest.TestCase):
    def test_read_train_data(self):

        # Initialize the function to test
        dataset = DataLoader().read_train_data()

        # Assert that the dataset is not empty
        self.assertNotEqual(len(list(dataset)), 0)

        # Assert that the data in the dataset is in the correct format
        for x, y in dataset:
            # Assert that x and y are tensors
            self.assertIsInstance(x, tf.Tensor)
            self.assertIsInstance(y, tf.Tensor)

            # Assert that x and y have the correct shape
            self.assertEqual(x.shape, (12, 512, 512, 3))
            self.assertEqual(y.shape, (12, 512, 512, 1))

            # Assert that x and y are normalized
            self.assertLessEqual(tf.reduce_max(x), 1)
            self.assertGreaterEqual(tf.reduce_min(x), 0)
            self.assertLessEqual(tf.reduce_max(y), 1)
            self.assertGreaterEqual(tf.reduce_min(y), 0)