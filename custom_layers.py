import tensorflow as tf
import keras
from keras import layers
import numpy as np

class OneHotWithMasking(layers.Layer):
    """
    A custom layer that performs one-hot encoding and supports masking.
    
    It identifies a specific value (by default, 0) as the padding token
    and generates a mask to ignore it in subsequent layers.
    """
    def __init__(self, depth, mask_token=0, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.mask_token = mask_token
        # This tells Keras that the layer supports masking
        self.supports_masking = True

    def call(self, inputs):
        # Perform the one-hot encoding
        inputs = tf.cast(inputs, dtype=tf.int32)
        one_hot_encoded = tf.one_hot(inputs, depth=self.depth)
        
        # We explicitly multiply by the mask to ensure the output for
        # padded steps is a zero-vector. This can help with stability.
        mask = self.compute_mask(inputs)
        mask_expanded = tf.expand_dims(tf.cast(mask, dtype=tf.float32), axis=-1)
        
        return one_hot_encoded * mask_expanded

    def compute_mask(self, inputs, mask=None):
        # This is the core of the masking mechanism.
        # It creates a boolean mask that is True for valid tokens
        # and False for the padding token.
        return tf.not_equal(inputs, self.mask_token)

    def get_config(self):
        # Required for saving and loading the model
        config = super().get_config()
        config.update({
            "depth": self.depth,
            "mask_token": self.mask_token
        })
        return config
    
if __name__ == "__main__":
    # Define parameters
    vocab_size = 5
    input = tf.convert_to_tensor([[1, 2, 3, 4, 1, 2, 3, 0, 0, 0, 0]], dtype=tf.int32)

    one_hot = OneHotWithMasking(depth=vocab_size)
    output = one_hot(input)
    print(output)