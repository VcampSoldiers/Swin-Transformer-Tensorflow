import os
import requests
import tarfile
import glob

import tensorflow as tf

from models.swin_transformer import SwinTransformer


def build_model(config, load_pretrained=True, weights_type='imagenet-1k'):
    """ Build a model of which config is pre-set

    Downloads and loads pretrained weights from Github repository to load into Swin Transformer model
    if load_pretrained is True.

    Args:
        config: Predefined Swin Transformer model configuration.
        load_pretrained: If True, pretrained weights from official PyTorch Implementation are loaded into model
        weights_type: Dataset used for pretraining. Should be one of the following: 'imagenet_1k', 'imagenet_22k', 'imagenet_22kto1k'
    
    Returns:
        A TensorFlow Keras model fitting the configurations given in config and pretrained weights, depending
        on whether or not load_pretrained is True or False.
    """

    # Build model
    swin_transformer = build_model_with_config(config)

    if not load_pretrained:
        return swin_transformer

    # Prepare pretrained weights to download
    weights_link = f"https://github.com/VcampSoldiers/Swin-Transformer-Tensorflow/releases/download/v1.0/{config.MODEL.NAME}_{weights_type[9:]}.tar.gz"

    # Designate path for pretrained weights
    weights_folder = './weights'
    weights_dir_path = f'{weights_folder}/{config.MODEL.NAME}_{weights_type}'
    weights_tgz_path = weights_dir_path + '.tar.gz'

    # Download pretrained weights and load them into the model
    try:
        if os.path.exists(weights_dir_path):
            print('Pretrained weights file already exists. Skipping download...')
        else:
            if not os.path.isdir(weights_folder):
                os.mkdir(weights_folder)
            with open(weights_tgz_path, 'wb') as file:
                response = requests.get(weights_link)
                file.write(response.content)
            tar = tarfile.open(weights_tgz_path)
            tar.extractall(weights_dir_path)
        ckpt_filename = glob.glob(os.path.join(weights_dir_path, '*.ckpt.data-00000-of-00001'))[0].split('/')[-1].split('.')[0]
    except Exception as e:
        print(f'There is no matching pretrained weights for model: {config.MODEL.NAME}, dataset: {weights_type}')
        print('Skipping pretrained weights load...')
    else:
        input_shape = (1, config.MODEL.SWIN.IN_CHANS, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
        # Compile a model to load weights
        swin_transformer(tf.zeros(input_shape), training=False)
        swin_transformer.load_weights(os.path.join(weights_dir_path, ckpt_filename+'.ckpt'))
    finally:
        if os.path.isfile(weights_tgz_path):
            os.remove(weights_tgz_path)

    return swin_transformer


def build_model_with_config(config):
    model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN.EMBED_DIM,
                        depths=config.MODEL.SWIN.DEPTHS,
                        num_heads=config.MODEL.SWIN.NUM_HEADS,
                        window_size=config.MODEL.SWIN.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                        qk_scale=config.MODEL.SWIN.QK_SCALE,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN.APE,
                        patch_norm=config.MODEL.SWIN.PATCH_NORM)
    return model