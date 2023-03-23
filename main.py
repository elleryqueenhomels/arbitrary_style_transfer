# Demo - train the style transfer network & use it to generate an image

import argparse

from train import train
from infer import stylize
from utils import list_images

parser = argparse.ArgumentParser()

parser.add_argument("--is_training", type=bool, default=False, help="Using training mode, default value is False")
parser.add_argument("--training_content_dir", default="../MS_COCO", help="Content image directory for training")
parser.add_argument("--training_style_dir", default="../WikiArt", help="Style image directory for training")
parser.add_argument("--encoder_weights_path", default="vgg19_normalised.npz", help="Encoder weights file path")
parser.add_argument("--logging_period", type=int, default=20, help="Logging period, default value is 20")
parser.add_argument("--inferring_content_dir", default="images/content", help="Content image directory for inferring")
parser.add_argument("--inferring_style_dir", default="images/style", help="Style image directory for inferring")
parser.add_argument("--output_dir", default="outputs", help="Output directory for inferring")

args = parser.parse_args()

# for training
STYLE_WEIGHTS = [2.0]
MODEL_SAVE_PATHS = [
    'models/style_weight_2e0.ckpt',
]

TRAINING_CONTENT_DIR = args.training_content_dir
TRAINING_STYLE_DIR = args.training_style_dir
ENCODER_WEIGHTS_PATH = args.encoder_weights_path
LOGGING_PERIOD = args.logging_period

# for inferring (stylize)
INFERRING_CONTENT_DIR = args.inferring_content_dir
INFERRING_STYLE_DIR = args.inferring_style_dir
OUTPUTS_DIR = args.output_dir


def main():

    if args.is_training:

        content_imgs_path = list_images(TRAINING_CONTENT_DIR)
        style_imgs_path   = list_images(TRAINING_STYLE_DIR)

        for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
            print('\n>>> Begin to train the network with the style weight: %.2f\n' % style_weight)

            train(style_weight, content_imgs_path, style_imgs_path, ENCODER_WEIGHTS_PATH, 
                  model_save_path, logging_period=LOGGING_PERIOD, debug=True)

        print('\n>>> Successfully! Done all training...\n')

    else:

        content_imgs_path = list_images(INFERRING_CONTENT_DIR)
        style_imgs_path   = list_images(INFERRING_STYLE_DIR)

        for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
            print('\n>>> Begin to stylize images with style weight: %.2f\n' % style_weight)

            stylize(content_imgs_path, style_imgs_path, OUTPUTS_DIR, 
                    ENCODER_WEIGHTS_PATH, model_save_path, 
                    suffix='-' + str(style_weight))

        print('\n>>> Successfully! Done all stylizing...\n')


if __name__ == '__main__':
    main()
