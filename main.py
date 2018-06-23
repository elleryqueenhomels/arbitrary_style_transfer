# Demo - train the style transfer network & use it to generate an image

from __future__ import print_function

from train import train
from infer import stylize
from utils import list_images


IS_TRAINING = False

# for training
CONTENT_IMGS_DIR = '../MS_COCO'
STYLE_IMGS_DIR = '../WikiArt'
ENCODER_WEIGHTS_PATH = './vgg19_normalised.npz'

STYLE_WEIGHTS = [2.0]
MODEL_SAVE_PATHS = [
    'models/style_weight_2e0.ckpt',
]

# for inferring (stylize)
CONTENTS_DIR = './images/content'
STYLES_DIR = './images/style/'
OUTPUT_DIR = './outputs'
STYLES = [
    'cat', 'mosaic', 'escher_sphere',
    'lion', 'udnie', 'woman_matisse',
]


def main():

    if IS_TRAINING:

        content_imgs_path = list_images(CONTENT_IMGS_DIR)
        style_imgs_path   = list_images(STYLE_IMGS_DIR)

        for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
            print('\nBegin to train the network with the style weight: %.2f ...\n' % style_weight)

            train(style_weight, content_imgs_path, style_imgs_path, ENCODER_WEIGHTS_PATH, 
                  model_save_path, logging_period=LOGGING_PERIOD, debug=True)

        print('\n>>> Successfully! Done all training...\n')

    else:

        contents_path = list_images(CONTENTS_DIR)

        for style_name in STYLES:
            print('\nUse "%s.jpg" as style to generate images:' % style_name)

            for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
                print('\nBegin to generate images with the style weight: %.2f ...\n' % style_weight)

                style_path = STYLES_DIR + style_name + '.jpg'
                generated_images = stylize(contents_path, style_path, ENCODER_WEIGHTS_PATH, model_save_path, 
                    output_path=OUTPUT_DIR, prefix=style_name + '-', suffix='-' + str(style_weight))


if __name__ == '__main__':
    main()

