# -*- coding: utf-8 -*-
import click
import logging
import cv2
import os
import shutil
import json
import numpy as np
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from process_image import generate_subsections

logger = logging.getLogger(__name__)

@click.command()

@click.option('--input_filepath', prompt=False, type=click.Path())
@click.option('--output_filepath', prompt=False, type=click.Path())
@click.option('--seed', prompt=True, type=int)
@click.option('--number_per_real', prompt=True, type=int)
@click.option('--number_per_fake', prompt=True, type=int)
@click.option('--width', prompt=True, type=int)
@click.option('--height', prompt=True, type=int)
@click.option('--target_width', prompt=False, type=int)
@click.option('--target_height', prompt=False, type=int)
@click.option('--xrotation', prompt=True, type=int)
@click.option('--yrotation', prompt=True, type=int)
@click.option('--zrotation', prompt=True, type=int)
def main(input_filepath, output_filepath, seed, width, height, target_width, target_height, xrotation, yrotation, zrotation, number_per_real, number_per_fake):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger.info('making final data set from raw data')

    # Set default arguments if not provided
    if not target_height:
        target_height = None
    if not target_width:
        target_width = None
    else:
        logger.info('target_width, target_height): {} x {}'.format(target_width, target_height))

    if not input_filepath:
        input_filepath = "./data/raw/"

    generate_interim(seed,
                     width,
                     height,
                     input_filepath,
                     output_filepath,
                     target_height,
                     target_width,
                     xrotation,
                     yrotation,
                     zrotation,
                     real_prefix="real",
                     fake_prefix="fake",
                     number_real=number_per_real,
                     number_fake=number_per_fake)
    cleanup()


def generate_interim(seed, width, height, input_filepath, output_filepath, target_width, target_height, xrotation, yrotation, zrotation, real_prefix, fake_prefix, number_real, number_fake):
    # NOTE: Could make a lot more DRY but this is clearer and this is a 
    # single use util function
    interim_directory = "./data/interim/"
    #  make fake, real subdirectories within iterim so we can
    # apply differetn parametrizations generate subsections w/o thinking
    fake_interim_directory = Path("./data/interim/fake/")
    fake_interim_directory.mkdir(parents=True, exist_ok=True)
    real_interim_directory = Path("./data/interim/real/")
    real_interim_directory.mkdir(parents=True, exist_ok=True)
    
    # Then we copy each type of image into its respective directory ...
    for prefix, directory in [(fake_prefix+'*', fake_interim_directory),
                              (real_prefix+'*', real_interim_directory)]:
        for image in Path(input_filepath).glob(prefix):
            shutil.copy(image, str(directory))

    #  ... generate subsections against both
    for the_source_path, number_of_images in [(str(fake_interim_directory), number_fake),
                                              (str(real_interim_directory), number_real)]:
        logger.info('... generating {} image(s) each in for {}'.format(number_of_images, the_source_path))
        generate_subsections(seed,
                             number_of_images,
                             width,
                             height,
                             the_source_path+'/',
                             interim_directory,
                             target_height,
                             target_width,
                             (xrotation, yrotation, zrotation))
    
    generate_processed(target_height, target_width)

def load_image(img_path, shape=None):
    img = cv2.imread(img_path, flags=1)
    if shape is not None:
        img = cv2.resize(img, shape)

    return img

def generate_processed(target_height,
                       target_width,
                       interim_directory="./data/interim/",
                       processed_directory="./data/processed/"):
    number_of_images = len(list(Path('./data/interim/').glob('*.jpg')))

    # an RGB array for each image, of target height and width
    image_array = np.zeros((number_of_images, target_height, target_width, 3))
    class_array = np.zeros(number_of_images)
    filename_array = []
    for index, the_image in enumerate(Path('./data/interim/').glob('*.jpg')):
        image_name = the_image.name
        label = 0 if image_name.split('_')[0] == 'real' else -1
        class_array[index] = label
        image_array[index, :] = load_image(str(the_image))
        filename_array.append(image_name)
        logger.info('\t... added in {} ({})'.format(image_name, label))

    image_array.dump(processed_directory+'images.npy')
    class_array.dump(processed_directory+'classes.npy')
    with open(processed_directory+"filenames.json", "w") as obj:
        json.dump(filename_array, obj)

    #  can read back in with 
    # image_array = np.load('./data/processed/*.npy', allow_pickle=True) 


def cleanup():
    # These paths should be in a config file
    # remove the fake, real directories now that we have subsections generated
    interim_directory = "./data/interim/"
    #  make fake, real subdirectories within iterim so we can
    # apply differetn parametrizations generate subsections w/o thinking
    fake_interim_directory = Path("./data/interim/fake/")
    fake_interim_directory.mkdir(parents=True, exist_ok=True)
    real_interim_directory = Path("./data/interim/real/")
    real_interim_directory.mkdir(parents=True, exist_ok=True)

    for directory in [fake_interim_directory, real_interim_directory]:
        for the_file in directory.glob('*'):
            the_file.unlink()
        directory.rmdir() 

    interim_directory = Path("./data/interim/")
    interim_directory.mkdir(parents=True, exist_ok=True)
    for the_file in interim_directory.glob('*'):
        the_file.unlink()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
