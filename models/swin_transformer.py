import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
)
from tensorflow.keras import Model, Layer, Sequential


class Mlp(Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.nn.gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Dense(hidden_features)
        self.act = act_layer()
        self.fc2 = Dense(out_features)
        self.drop = Dropout(drop)

    def call(self, x):
        assert x.shape[-1] == self.in_features
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@tf.function
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C]) 
    # TODO contiguous memory access?
    windows = tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2, 4, 5]), [-1, window_size, window_size, C])
    return windows


@tf.function
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, -1])
    x = tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2, 4, 5]), [B, H, W, -1])
    return x
