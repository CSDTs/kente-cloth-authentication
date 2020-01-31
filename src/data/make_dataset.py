# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from process_image import generate_subsections

@click.command()


@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--seed', prompt=True, type=int)
@click.option('--n', prompt=True, type=int)
@click.option('--width', prompt=True, type=int)
@click.option('--height', prompt=True, type=int)
@click.option('--xrotation', prompt=True, type=int)
@click.option('--yrotation', prompt=True, type=int)
@click.option('--zrotation', prompt=True, type=int)

def main(input_filepath, output_filepath, seed, n, width, height, xrotation, yrotation, zrotation):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    generate_subsections(seed, n, width, width, input_filepath, output_filepath, (xrotation, yrotation, zrotation))
    



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
