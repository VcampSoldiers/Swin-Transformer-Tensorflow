import tensorflow as tf

from config import get_config_tiny
from models.build import build_model

if __name__ == "__main__":
    tf.debugging.set_log_device_placement(False)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    config = get_config_tiny()

    swin_transformer = build_model(config)
    swin_transformer(tf.zeros([1, 3, 224, 224]))
    swin_transformer.load_weights('./tf_weights/swin_tiny_patch4_window7_224.tf')

    print(swin_transformer(tf.zeros([1, 3, 224, 224])))