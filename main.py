import tensorflow as tf

from config import get_config_tiny, update_config_from_file
from models.build import build_model, build_model_with_config

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

    swin_transformer = build_model('swin_tiny_224')
    print(swin_transformer(tf.zeros([1, 3, 224, 224])))