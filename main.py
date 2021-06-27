import argparse

import tensorflow as tf

from config import get_config
from models.build import build_model

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer weight conversion from PyTorch to TensorFlow', add_help=False)

    parser.add_argument(
        '--cfg',
        type=str,
        metavar="FILE",
        help='path to config file',
        default="configs/swin_tiny_patch4_window7_224.yaml"
    )
    parser.add_argument(
        '--include_top',
        type=int,
        help='Whether or not to include model head',
        choices={0, 1},
        default=1,
    )
    parser.add_argument(
        '--resume',
        type=int,
        help='Whether or not to resume training from pretrained weights',
        choices={0, 1},
        default=1,
    )
    parser.add_argument(
        '--weights_type',
        type=str,
        help='Type of pretrained weight file to load including number of classes',
        choices={"imagenet_1k", "imagenet_22k", "imagenet_22kto1k"},
        default="imagenet_1k",
    )

    args = parser.parse_args()
    config = get_config(args, include_top=bool(args.include_top))

    return args, config

def main(args, config):
    swin_transformer = build_model(config, load_pretrained=bool(args.resume), weights_type=args.weights_type)
    print(
        swin_transformer(
            tf.zeros([
                1,
                config.MODEL.SWIN.IN_CHANS,
                config.DATA.IMG_SIZE,
                config.DATA.IMG_SIZE
            ])
        )
    )


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

    args, config = parse_option()
    main(args, config)