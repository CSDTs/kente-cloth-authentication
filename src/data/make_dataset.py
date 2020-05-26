# -*- coding: utf-8 -*-
import click
import logging
import cv2
import os
import shutil
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from process_image import generate_subsections
from dotenv import find_dotenv, load_dotenv
from itertools import chain
import pandas as pd
import random

logger = logging.getLogger(__name__)


@click.group()
@click.option('--input_filepath', type=click.Path())
@click.option('--output_filepath', type=click.Path())
@click.option('--seed', type=int)
# @click.option('--number_per_real', type=int)
@click.option('--create_balanced_real_fake', type=int)
@click.option('--width', type=int)
# @click.option('--number_per_fake', type=int)
@click.option('--height', type=int)
@click.option('--target_width', type=int)
@click.option('--target_height', type=int)
@click.option('--xrotation', type=int)
@click.option('--yrotation', type=int)
@click.option('--zrotation', type=int)
@click.option('--whole_cloth_stratify', nargs=3, type=int)
def main(input_filepath, output_filepath, seed, width, height, target_width, target_height, xrotation, yrotation,
         zrotation, create_balanced_real_fake, whole_cloth_stratify):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    # logger.info('making final data set from raw data')

    # # Set default arguments if not provided
    # if not target_height:
    #     target_height = None
    # if not target_width:
    #     target_width = None
    # else:
    #     logger.info('target_width, target_height): {} x {}'.format(target_width, target_height))

    # if not input_filepath:
    #     input_filepath = "./data/raw/"

    # generate_interim(seed,
    #                  width,
    #                  height,
    #                  input_filepath,
    #                  output_filepath,
    #                  target_height,
    #                  target_width,
    #                  xrotation,
    #                  yrotation,
    #                  zrotation,
    #                  real_prefix="real",
    #                  fake_prefix="fake",
    #                  number_real=number_per_real,
    #                  number_fake=number_per_fake)

    # generate_processed(target_height, target_width)
    # cleanup()

def load_image(img_path, shape=None):
    img = cv2.imread(img_path, flags=1)
    if shape is not None:
        img = cv2.resize(img, shape)

    return img

def get_inlier_outlier_groups(num_samples, inlier_groups, outlier_groups):
    """
    Get even number of unique inlier and outlier groups
    """
    num_inliers = int(num_samples / 2)
    num_outliers = num_samples - num_inliers
    inlier_groups = random.sample(inlier_groups, num_inliers)
    outlier_groups = random.sample(outlier_groups, num_outliers)
    groups = inlier_groups + outlier_groups
    return groups

@main.command()
def generate_three_data_splits():
    """
    Currently just manually copy out
    """
    raise NotImplemented

@main.command()
@click.option('--input_filepath', '-i', type=click.Path())
@click.option('--output_filepath', '-o', type=click.Path())
@click.option('--seed', type=int)
# @click.option('--number_per_real', type=int)
# @click.option('--number_per_fake', type=int)
@click.option('--create_balanced_real_fake', type=int)
@click.option('--real_prefix', default="real")
@click.option('--fake_prefix', default="fake")
@click.option('--width', type=int)
@click.option('--height', type=int)
@click.option('--target_width', default=233, type=int)
@click.option('--target_height', default=233, type=int)
@click.option('--xrotation', type=int)
@click.option('--yrotation', type=int)
@click.option('--zrotation', type=int)
def makeinterim(seed, width, height, input_filepath, output_filepath, target_width, target_height, xrotation, yrotation,
                zrotation, real_prefix, fake_prefix, create_balanced_real_fake):
    # NOTE: Could make a lot more DRY but this is clearer and this is a
    # single use util function
    interim_directory = "./data/interim/"
    #  make fake, real subdirectories within iterim so we can
    # apply differetn parametrizations generate subsections w/o thinking
    fake_interim_directory = Path("./data/interim/fake/")
    fake_interim_directory.mkdir(parents=True, exist_ok=True)
    real_interim_directory = Path("./data/interim/real/")
    real_interim_directory.mkdir(parents=True, exist_ok=True)

    # Creating number_per_fake and number_per_real

    real = 0
    fake = 0

    for filename in os.listdir('./data/raw'):

        if filename.startswith('fake'):
            fake = fake + 1

        elif filename.startswith('real'):
            real = real + 1

    number_per_real = int(create_balanced_real_fake / real)
    number_per_fake = int(create_balanced_real_fake / fake)

    if not Path(input_filepath).exists():
        print(f"Input path provided is: {input_filepath}")
        return
        # raise FileNotFoundError, "Input file path does not exist!"

    # Then we copy each type of image into its respective directory ...
    for prefix, directory in [(fake_prefix + '*', fake_interim_directory),
                              (real_prefix + '*', real_interim_directory)]:
        for image in Path(input_filepath).glob(prefix):
            shutil.copy(image, str(directory))

    #  ... generate subsections against both
    for the_source_path, number_of_images in [(str(fake_interim_directory), number_per_fake),
                                              (str(real_interim_directory), number_per_real)]:
        logger.info('... generating {} image(s) each in for {}'.format(number_of_images, the_source_path))
        generate_subsections(seed,
                             number_of_images,
                             width,
                             height,
                             the_source_path + '/',
                             interim_directory,
                             target_height,
                             target_width,
                             (xrotation, yrotation, zrotation))

    # ... finally, we clean up the interim directories
    for directory in [fake_interim_directory, real_interim_directory]:
        for the_file in directory.glob('*'):
            the_file.unlink()
        directory.rmdir()


@main.command()
@click.option('--interim_directory', '-i', type=str, default="./data/interim/")
@click.option('--processed_directory', '-o', type=str, default="./data/processed/")
@click.option('--evaluation_directory', '-e', type=str, default="./data/processed/evaluation/")
@click.option('--training_directory', '-t', type=str, default="./data/processed/training/")
@click.option('--validation_directory', '-v', type=str, default="./data/processed/validation/")
@click.option('--number_outlier_test_groups', '-notg', type=int, default=1)
@click.option('--number_inlier_test_groups', '-nitg', type=int, default=4)
@click.option('--whole_cloth_stratify', '-wcs', nargs=3, type=int)
@click.option('--inlier', '-inl', default=1, type=int)
@click.option('--outlier', '-out', default=-1, type=int)
def makeprocessed(number_outlier_test_groups,
                  number_inlier_test_groups,
                  interim_directory,
                  processed_directory,
                  evaluation_directory,
                  training_directory,
                  validation_directory,
                  whole_cloth_stratify,
                  preserve_shuffle=False,
                  inlier=1,
                  outlier=-1):
    #  This function copies out interim images into training, evaluation and validation
    # training datasets into processed.

    interim_directory = Path(interim_directory)
    evaluation_directory = Path(evaluation_directory)
    evaluation_directory.mkdir(exist_ok=True)

    training_directory = Path(training_directory)
    training_directory.mkdir(exist_ok=True)

    validation_directory = Path(validation_directory)
    validation_directory.mkdir(exist_ok=True)

    number_of_images = len(list(interim_directory.glob('*.jpg')))
    y = np.full((number_of_images,), inlier)
    the_groups = []
    the_file_paths = []
    for index, file_name in enumerate(interim_directory.glob('*.jpg')):
        label = file_name.name.split('_')[0]
        group = file_name.name.rsplit('_',1)[0]
        if label == 'fake':
            y[index] = outlier
        the_groups.append(group)
        the_file_paths.append(file_name)

    sampling_frame = \
        pd.DataFrame(
            {"group": the_groups,
             "file_path": the_file_paths,
             "y": y}
        )

    if len(whole_cloth_stratify) == 0:
        #  ... we split this up into evaluation (testing) and then the remaining we
        # split into validation and training. We focus top down on how
        # many inlier and outlier groups the evaluation set should have, randomly chose those
        # then split the remaining instances 50%/50% to create the validation, training
        # groups.
        #
        # This assumes that the evaluation groups represent a balanced set; the remaining
        # data is stratified so that's much more balanced by design. If the original
        # data is balanced (which is is in my case) then everything will be roughly balanced
        inlier_test_groups = \
            set(
                sampling_frame.sample(frac=1, random_state=42) \
                    .query(f'y=={inlier}') \
                    .group \
                    .unique()[:number_inlier_test_groups]
            )

        outlier_test_groups = \
            set(
                sampling_frame.sample(frac=1, random_state=42) \
                    .query(f'y=={outlier}') \
                    .group \
                    .unique()[:number_outlier_test_groups]
            )
        test_groups = inlier_test_groups | outlier_test_groups

        # ... finally we split the remaining data into balanced training and validation
        validation_training_df = \
            sampling_frame.query('group not in @test_groups')
        validation_indices, training_indices = \
            next(
                StratifiedShuffleSplit(random_state=42,
                                       n_splits=1,
                                       test_size=0.5).split(
                    validation_training_df.drop('y', axis=1),
                    validation_training_df.y)
            )

    else:
        num_train, num_validation, num_eval = whole_cloth_stratify
        random.seed(6)

        inlier_groups = list(sampling_frame.query(f'y=={inlier}').group.unique())
        outlier_groups = list(sampling_frame.query(f'y=={outlier}').group.unique())
        num_unique = len(sampling_frame.group.unique())

        if num_eval + num_train + num_validation > num_unique:
            raise ValueError("whole_cloth_stratify: sum of arguments is greater than number of unique images")

        #If num_eval is -1 then use 1/3rd of unique images
        if num_eval < 0:
            num_eval = round(num_unique / 3)

        #
        test_groups = get_inlier_outlier_groups(num_eval, inlier_groups, outlier_groups)
        inlier_groups = [i for i in inlier_groups if i not in test_groups]
        outlier_groups = [i for i in outlier_groups if i not in test_groups]

        validation_training_df = sampling_frame.query('group not in @test_groups').reset_index(drop=True)

        if num_train > 0 and num_validation > 0:
            # Randomly pick num_train unique images for training set
            # then randomly pick num_validation unique images for validation set
            train_groups = get_inlier_outlier_groups(num_train, inlier_groups, outlier_groups)
            inlier_groucps = [i for i in inlier_groups if i not in train_groups]
            outlier_groups = [i for i in outlier_groups if i not in train_groups]

            training_indices = validation_training_df.query('group in @train_groups').index

            validation_groups = get_inlier_outlier_groups(num_validation, inlier_groups, outlier_groups)
            inlier_groups = [i for i in inlier_groups if i not in validation_groups]
            outlier_groups = [i for i in outlier_groups if i not in validation_groups]
            validation_indices = validation_training_df.query('group in @validation_groups').index

        elif num_train < 0 and num_validation < 0:
            # StratifiedShuffleSplit remaining images into training and validation
            validation_indices, training_indices = \
                next(
                    StratifiedShuffleSplit(random_state=42,
                                           n_splits=1,
                                           test_size=0.5).split(
                        validation_training_df.drop('y', axis=1),
                        validation_training_df.y)
                    )
        else:
            raise ValueError("whole_cloth_stratify: invalid values! Should be -1 or a positive integer")

    # Copy images from interim into training, validation and test data
    # folders based on the whole_cloth_stratify

    copy_to_directory = validation_directory
    validation_training_df.iloc[validation_indices] \
        .apply(lambda row: shutil.copy(
        str(row.file_path),
        str(copy_to_directory / row.file_path.name)),
               axis=1)

    copy_to_directory = training_directory
    validation_training_df.iloc[training_indices] \
        .apply(lambda row: shutil.copy(
        str(row.file_path),
        str(copy_to_directory / row.file_path.name)),
               axis=1)

    copy_to_directory = evaluation_directory
    sampling_frame.query('group in @test_groups') \
        .apply(lambda row: shutil.copy(
        str(row.file_path),
        str(copy_to_directory / row.file_path.name)),
               axis=1)


@main.command()
def cleanup():
    # These paths should be in a config file
    # remove the fake, real directories now that we have subsections generated
    interim_directory = "./data/interim/"
    #  make fake, real subdirectories within iterim so we can
    # apply differetn parametrizations generate subsections w/o thinking

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
