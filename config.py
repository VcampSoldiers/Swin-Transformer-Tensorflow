# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()

_C.DATA.IMG_SIZE = 224

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()

def get_config_tiny():
    config = _C.clone()

    config.defrost()

    config.MODEL.TYPE = 'swin'
    # Model name
    config.MODEL.NAME = 'swin_tiny_patch4_window7_224'
    # Number of classes, overwritten in data preparation
    config.MODEL.NUM_CLASSES = 1000
    # Dropout rate
    config.MODEL.DROP_RATE = 0.0
    # Drop path rate
    config.MODEL.DROP_PATH_RATE = 0.1

    # Swin Transformer parameters
    config.MODEL.SWIN = CN()
    config.MODEL.SWIN.PATCH_SIZE = 4
    config.MODEL.SWIN.IN_CHANS = 3
    config.MODEL.SWIN.EMBED_DIM = 96
    config.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    config.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    config.MODEL.SWIN.WINDOW_SIZE = 7
    config.MODEL.SWIN.MLP_RATIO = 4.
    config.MODEL.SWIN.QKV_BIAS = True
    config.MODEL.SWIN.QK_SCALE = None
    config.MODEL.SWIN.APE = False
    config.MODEL.SWIN.PATCH_NORM = True

    config.freeze()

    return config