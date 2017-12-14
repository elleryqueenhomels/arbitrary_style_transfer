# Arbitrary-Style-Transfer

Arbitrary-Style-Per-Model Fast Neural Style Transfer Method

## Description
Using an <b>Encoder-AdaIN-Decoder</b> architecture - Deep Convolutional Neural Network as a Style Transfer Network (STN) which can receive two arbitrary images as inputs (one as content, the other one as style) and output a generated image that recombines the content and spatial structure from the former and the style (color, texture) from the latter without re-training the network. The STN is trained using MS-COCO dataset (about 12.6GB) and WikiArt dataset (about 36GB).

This code is based on Huang et al. [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/pdf/1703.06868.pdf) *(ICCV 2017)*

![stn_overview](https://user-images.githubusercontent.com/13844740/33978899-d428bf2e-e0dc-11e7-9114-41b6fb8921a7.jpg)
System overview. Picture comes from Huang et al. original paper. The encoder is a fixed VGG-19 (up to relu4_1) which is pre-trained on ImageNet dataset for image classification. We train the decoder to invert the AdaIN output from feature spaces back to the image spaces.

## Prerequisites
- [Pre-trained VGG19 normalised network](https://s3.amazonaws.com/xunhuang-public/adain/vgg_normalised.t7) (MD5 `c637adfa9cee4b33b59c5a754883ba82`) <br/><b>I have provided a convertor in the `tool` folder. It can extract kernel and bias from the torch model file (.t7 format) and save them into a npz file which is easier to process via NumPy.</b> <br/><b>Or you can simply download my pre-processed file:</b> <br/>[Pre-trained VGG19 normalised network npz format](https://s3-us-west-2.amazonaws.com/wengaoye/vgg19_normalised.npz) (MD5 `c5c961738b134ffe206e0a552c728aea`)
- [Microsoft COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)
- [WikiArt dataset](https://www.kaggle.com/c/painter-by-numbers)

## Trained Model
You can download my trained model from [here](https://s3-us-west-2.amazonaws.com/wengaoye/arbitrary_style_model_style-weight-2e0.zip) which is trained with style weight equal to 2.0<br/>Or you can directly use `download_trained_model.sh` in the repo.

## Manual
- The main file `main.py` is a demo, which has already contained training procedure and inferring procedure (inferring means generating stylized images).<br />You can switch these two procedures by changing the flag `IS_TRAINING`.
- By default,<br />(1) The content images lie in the folder `"./images/content/"`<br />(2) The style images lie in the folder `"./images/style/"`<br />(3) The weights file of the pre-trained VGG-19 lies in the current working directory. (See `Prerequisites` above. By the way, `download_vgg19.sh` already takes care of this.)<br />(4) The MS-COCO images dataset for training lies in the folder `"../MS_COCO/"` (See `Prerequisites` above)<br />(5) The WikiArt images dataset for training lies in the folder `"../WikiArt/"` (See `Prerequisites` above)<br />(6) The checkpoint files of trained models lie in the folder `"./models/"` (You should create this folder manually before training.)<br />(7) After inferring procedure, the stylized images will be generated and output to the folder `"./outputs/"`
- For training, you should make sure (3), (4), (5) and (6) are prepared correctly.
- For inferring, you should make sure (1), (2), (3) and (6) are prepared correctly.
- Of course, you can organize all the files and folders as you want, and what you need to do is just modifying related parameters in the `main.py` file.

## Results
| style | output (generated image) |
| :----: | :----: |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/udnie_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/udnie-lance-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/escher_sphere_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/escher_sphere-lance-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/mosaic_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/mosaic-lance-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/cat_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/cat-lance-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/lion_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/lion-lance-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/woman_matisse_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/woman_matisse-lance-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/udnie_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/udnie-brad_pitt-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/escher_sphere_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/escher_sphere-brad_pitt-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/mosaic_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/mosaic-brad_pitt-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/udnie_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/udnie-chicago-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/escher_sphere_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/escher_sphere-chicago-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/cat_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/cat-chicago-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/lion_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/lion-chicago-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/woman_matisse_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/woman_matisse-chicago-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/escher_sphere_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/escher_sphere-karya-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/lion_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/lion-karya-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/escher_sphere_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/escher_sphere-stata-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/mosaic_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/mosaic-stata-2.0.jpg)  |
|![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/images/style_thumb/cat_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/outputs/cat-stata-2.0.jpg)  |

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

