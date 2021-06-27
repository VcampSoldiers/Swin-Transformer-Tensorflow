# Swin-Transformer-Tensorflow
A direct translation of the official PyTorch implementation of ["Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"](https://arxiv.org/abs/2103.14030) to TensorFlow 2.

The official Pytorch implementation can be found [here](https://github.com/microsoft/Swin-Transformer).

## Introduction:
![Swin Transformer Architecture Diagram](./images/swin-transformer.png)

**Swin Transformer** (the name `Swin` stands for **S**hifted **win**dow) is initially described in [arxiv](https://arxiv.org/abs/2103.14030), which capably serves as a
general-purpose backbone for computer vision. It is basically a hierarchical Transformer whose representation is
computed with shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention
computation to non-overlapping local windows while also allowing for cross-window connection.

Swin Transformer achieves strong performance on COCO object detection (`58.7 box AP` and `51.1 mask AP` on test-dev) and
ADE20K semantic segmentation (`53.5 mIoU` on val), surpassing previous models by a large margin.


## Usage:
### 1. Create a model with pretrained weights
```python
from models.build import build_model

swin_transformer = build_model(model_name='swin_tiny_224', load_pretrained=True, include_top=True, weights_type='imagenet-1k')
```
Possible options for `model_name` and `weights_type` are:  
|model_name|weights_type|
|----|----|
|swin_tiny_224|imagenet-1k|
|swin_small_224|imagenet-1k|
|swin_base_224|imagenet-1k|
|swin_base_384|imagenet-1k|
|swin_base_224|imagenet-22k|
|swin_base_384|imagenet-22k|
|swin_large_224|imagenet-22k|
|swin_large_384|imagenet-22k|

If want to create your own classification model, try:
```python
import tensorflow as tf

from models.build import build_model

swin_transformer = tf.keras.Sequential([
    build_model(model_name='swin_tiny_224', load_pretrained=True, include_top=False, weights_type='imagenet-1k'),
    tf.keras.layers.Dense(NUM_CLASSES)
)
```
**Model ouputs are logits, so don't forget to include softmax in training/inference!!**

### 2. Load your own model configs
You can easily overwrite model configs with custom yaml file:
```python
from config import get_config_tiny, update_config_from_file
from models.build import build_model, build_model_with_config

config = get_tiny_config()
config = update_config_from_file(config, PATH_TO_YAML_FILE)
swin = build_model_with_config(config)
```
The example yaml file is provided in `./configs` directory.

### 3. Convert PyTorch pretrained weights into Tensorflow checkpoints
We provide a python script with which we convert official PyTorch weights into Tensorflow checkpoints:
```bash
$ python3 load_weights.py --weights the_path_to_pytorch_weights --output the_path_to_output_tf_weights
```
## TODO:
- [x] Translate model code over to TensorFlow
- [x] Load PyTorch pretrained weights into TensorFlow model
- [ ] Write trainer code
- [ ] Reproduce results presented in paper
    - [ ] Object Detection
- [ ] Reproduce training efficiency of official code in TensorFlow

### Citations: 
```bibtex
@misc{liu2021swin,
      title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows}, 
      author={Ze Liu and Yutong Lin and Yue Cao and Han Hu and Yixuan Wei and Zheng Zhang and Stephen Lin and Baining Guo},
      year={2021},
      eprint={2103.14030},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```