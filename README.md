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


## TODO:
- [x] Translate model code over to TensorFlow
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