# Demo - train the style transfer network & use it to generate an image

from __future__ import print_function

from train import train
from generate import generate
from utils import list_images


IS_TRAINING = False

ENCODER_WEIGHTS_PATH = './vgg19_normalised.npz'

STYLE_WEIGHTS = [2.0]

MODEL_SAVE_PATHS = [
    'models/style_weight_2e0.ckpt',
]

STYLES = [
    'cat', 'mosaic', 'escher_sphere',
    'lion', 'udnie', 'woman_matisse',
]


def main():

    if IS_TRAINING:

        content_imgs_path = list_images('../MS_COCO') # path to training content dataset
        style_imgs_path   = list_images('../WikiArt') # path to training style dataset

        for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
            print('\nBegin to train the network with the style weight: %.2f ...\n' % style_weight)

            train(style_weight, content_imgs_path, style_imgs_path, ENCODER_WEIGHTS_PATH, model_save_path, debug=True)

            print('\nSuccessfully! Done training...\n')
    else:

        for style_name in STYLES:

            print('\nUse "%s.jpg" as style to generate images:' % style_name)

            for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
                print('\nBegin to generate images with the style weight: %.2f ...\n' % style_weight)

                contents_path = list_images('images/content')
                style_path    = 'images/style/' + style_name + '.jpg'
                output_save_path = 'outputs'

                generated_images = generate(contents_path, style_path, ENCODER_WEIGHTS_PATH, model_save_path, 
                    output_path=output_save_path, prefix=style_name + '-', suffix='-' + str(style_weight))

                print('\nlen(generated_images): %d\n' % len(generated_images))


if __name__ == '__main__':
    main()

