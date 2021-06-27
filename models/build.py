import os
import requests
import tarfile
import glob

import tensorflow as tf

from config import get_config_tiny
from models.swin_transformer import SwinTransformer


def build_model(model_name, load_pretrained=True, include_top=True, weights_type='imagenet-1k'):
    """ Build a model of which config is pre-set

    Args:
        model_name: Predefined Swin Transformer model name.
        load_pretrained: If True, load pretrained weights from official PyTorch Implementation
        weights_type: Dataset used for pretraining. It should be one of 'imagenet_1k', 'imagenet_22k', 'imagenet_22kto1k'
    """

    # build a model
    if model_name == 'swin_tiny_224':
        config = get_config_tiny(include_top)
        swin_transformer = build_model_with_config(config)
    else:
        raise NotImplementedError(f'Model {model_name} is either not supported or not implemented yet.')

    if not load_pretrained:
        return swin_transformer

    # pretrained weights download link
    weights_link = f"https://github.com/VcampSoldiers/Swin-Transformer-Tensorflow/releases/download/v1.0/{model_name}_{weights_type[9:]}.tar.gz"

    # paths for pretrained weights
    weights_folder = './weights'
    weights_dir_path = f'{weights_folder}/{model_name}_{weights_type}'
    weights_tgz_path = weights_dir_path + '.tar.gz'

    # download pretrained weights and load it into model
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
        print(f'There is no matching pretrained weights for model: {model_name}, dataset: {weights_type}')
        print('Skipping pretrained weights load...')
    else:
        input_shape = (1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
        # compile a model to load weights
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