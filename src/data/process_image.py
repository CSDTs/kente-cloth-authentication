import random
import cv2
from image_transform import ImageTransformer, save_image, load_image
import fnmatch
import os
import re


def find_files(pattern, directory='.'):
    """
    Finds the images by pattern and returns the list

    Given a pattern and directory, goes through the directory and returns
    the images based on the given pattern. Igores case.

    Parameters
    ----------
    pattern : str
        Regex pattern for desired image
    directory : str
       Directory path containing the images

    Returns
    -------
    arr
        Array of names that match the given pattern.

    """
    rule = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
    return [name for name in os.listdir(directory) if rule.match(name)]

def generate_subsections(seed, N, W, H, input_filepath, output_filepath, xyz = [0,0,0]):
    """
    Usage
    ----------
    Change main function with desired arguments
    Then
    from process_image import generate_subsections

    Processes desired directory of images to create random subsections.

    Parameters
    ----------
    seed              : Seed value that allows this process to be deterministic
    N                 : Number of sub image/sub sections desired
    W                 : Width of subsection
    H                 : Height of
    input_filepath    : Location of images to be processed
    output_filepath   : Location of subsections to be saved
    xyz               : Tuple containing desired rotation.
                    Default is (0,0,0)

    Output
    ----------
    subsections       : Random subsections of each image

    Note: For the rotation, opencv's warpperspective was used. The bordertype is set
    to reflect instead of having a background fill. Also, in ImageTransformer, you
    are able to set an offset for each image.
    """

    #  Contains random seed that allows this process to be deterministic
    random.seed(seed)

    # To contain the list of images in given directory
    image_list = []

    # Creates new ImageTransformer for each image in directory.
    # Appends tuple of ImageTranformer and image name for later use
    for filename in find_files('*.jpg', directory=input_filepath):
        img = ImageTransformer(input_filepath + filename, None)
        img_name =filename.split('.')
        image_list.append((img, img_name[0]))

    # For each image, generate N number of subsections, randomly located on
    # the image. Each subsection is has a width of W, and a height of H.
    # Each image will be rotated first, before cropping.
    # Output will be saved to given file path.
    # NameOfImage_NumberOfSubsection.jpg
    for img in image_list:
        for x in range(0, N):
            # Offset size of subsection to avoid grabbing incomplete image.
            left = random.randint(0, img[0].width-W)
            upper = random.randint(0, img[0].height-H)
            right = left + W
            lower = upper + H

            #Rotate image given x,y,z
            rotated_img = img[0].rotate_along_axis(theta = xyz[0], phi=xyz[1], gamma=xyz[2])

            # Crop image based of subsection dimensions
            cropped_img = rotated_img[upper:lower, left:right]

            # Save image to output filepath
            save_image(output_filepath + img[1] + '_%d.jpg' % (x), cropped_img)


#Example
# generate_subsections(3123412,12,300,300,"../../data/raw/","../../data/processed/",(0,0,0))

