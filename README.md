# Arbitrary-Style-Transfer

Arbitrary-Style-Per-Model Fast Neural Style Transfer Method

## Description
Using an <b>Encoder-AdaIN-Decoder</b> architecture Deep Convolutional Neural Network as the Style Transfer Network which can take two arbitrary images as input (one as content, the orther one as style) and output a generated image that holds the content and structure from the former and the style from the latter without re-training the network.
The network is trained over Microsoft COCO dataset and WikiArt dataset.

This code is based on Huang et al. [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/pdf/1703.06868.pdf) *(ICCV 2017)*

## Prerequisites
- [Pre-trained VGG19 network](https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19_normalized.pkl) (MD5 `8ee3263992981a1d26e73b3ca028a123`)
- [Microsoft COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)
- [WikiArt dataset](https://www.kaggle.com/c/painter-by-numbers)

## My Running Environment
<b>Hardware</b>
- CPU: Intel® Core™ i9-7900X (3.30GHz x 10 cores, 20 threads)
- GPU: NVIDIA® Titan Xp (Architecture: Pascal, Frame buffer: 12GB)
- Memory: 32GB DDR4

<b>Operating System</b>
- ubuntu 16.04.03 LTS

<b>Software</b>
- Python 3.6.2
- NumPy 1.13.1
- TensorFlow 1.3.0
- SciPy 0.19.1
- CUDA 8.0.61
- cuDNN 6.0.21

## References
- The Encoder which is implemented with first few layers(up to relu4_1) of a pre-trained VGG-19 is based on [Anish Athalye's vgg.py](https://github.com/anishathalye/neural-style/blob/master/vgg.py)

## Citation
```
  @misc{ye2017arbitrarystyletransfer,
    author = {Wengao Ye},
    title = {Arbitrary Style Transfer},
    year = {2017},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/elleryqueenhomels/arbitrary_style_transfer}}
  }
```

