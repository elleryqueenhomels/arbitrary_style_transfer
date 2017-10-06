# Demo - train the style transfer network & use it to generate an image

from __future__ import print_function

from train import train
from generate import generate
from utils import list_images


IS_TRAINING = True

ENCODER_WEIGHTS_PATH = './vgg19_normalised.npz'

STYLE_WEIGHTS = [1.0, 0.5, 10.0, 0.1, 0.01]

MODEL_SAVE_PATHS = [
    'models/style_weight_1e0.ckpt',
    'models/style_weight_5e-1.ckpt',
    'models/style_weight_1e1.ckpt',
    'models/style_weight_1e-1.ckpt',
    'models/style_weight_1e-2.ckpt',
]

STYLES = [
    'wave', 'udnie', 'escher_sphere', 'flower', 
    'scream', 'denoised_starry', 'rain_princess', 
    'woman_matisse', 'mosaic'
]


def main():

    if IS_TRAINING:

        content_imgs_path = list_images('D:/ImageDatabase/Microsoft_COCO2014/train2014')  # path to training content dataset
        style_imgs_path = list_images('D:/ImageDatabase/WikiArt_database/all')  # path to training style dataset

        # content_imgs_path = list_images('D:/ImageDatabase/Microsoft_COCO2014/train2014')  # path to training content dataset
        # style_imgs_path = list_images('D:/ImageDatabase/WikiArt_database/train_1')  # path to training style dataset

        # content_imgs_path = list_images('D:/ImageDatabase/train_data_temp/MS_COCO_1000')  # path to training content dataset
        # style_imgs_path = list_images('D:/ImageDatabase/train_data_temp/WikiArt_1000')  # path to training style dataset

        for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
            print('\nBegin to train the network with the style weight: %.2f ...\n' % style_weight)

            train(style_weight, content_imgs_path, style_imgs_path, ENCODER_WEIGHTS_PATH, model_save_path, debug=True)

            print('\nSuccessfully! Done training...\n')
    else:

        for style_name in STYLES:

            print('\nUse "%s.jpg" as style to generate images:')

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

