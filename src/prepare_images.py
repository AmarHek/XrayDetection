import os

import cv2
import numpy as np
import pydicom
import tqdm


def shift_pixel_range(image, old_max, new_max=1.0):
    """
    Shifts the pixel values of an image to a new range.

    :param image: The image as a numpy array.
    :param old_max: The maximum value of the pixel values.
    :param new_max: The new maximum value of the pixel values.
    :return: The image with pixel values in the new range.
    """

    image_rescaled = np.rint((image / old_max) * new_max)
    return image_rescaled.astype(np.uint8)


def resize_image_with_fixed_aspect_ratio(image, new_size=640):
    """
    Resizes an image to a fixed size while maintaining the aspect ratio.
    The image is resized to the maximum size of either the width or height.
    The longer side is inferred from the image shape.

    :param image: The image as a numpy array.
    :param new_size: The new size of the longer side.
    :return: resized image
    """

    # get the image shape
    height, width = image.shape[:2]

    # determine the longer side
    if height > width:
        new_height = new_size
        new_width = round(width * new_size / height)
    elif width > height:
        new_width = new_size
        new_height = round(height * new_size / width)
    else:
        new_height = new_size
        new_width = new_size

    return cv2.resize(image, (new_width, new_height))


def image_preprocessing_pipeline(dicom_path, png_path):
    """
    Converts a dicom image to a png image.

    :param dicom_path: The path to the dicom image.
    :param png_path: The path to save the png image.
    """

    # extract the pixel array
    dicom = pydicom.dcmread(dicom_path)
    image = dicom.pixel_array

    # shift the pixel values to [0, 255] based on the bit depth
    old_max = 2 ** dicom.BitsStored - 1
    image = shift_pixel_range(image, old_max=old_max, new_max=255)

    # resize the image to a fixed size
    image = resize_image_with_fixed_aspect_ratio(image, new_size=640)

    # convert to rgb
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    cv2.imwrite(png_path, image)


def preprocess_dataset(dicom_root, png_root):
    """
    Preprocesses a dataset of dicom images to png images.
    :param dicom_root: root directory containing the dicom images.
    :param png_root: directory to save the png images.
    """

    # check if the dicom root directory exists
    if not os.path.exists(dicom_root):
        raise FileNotFoundError(f"Directory not found: {dicom_root}")

    # create the png root directory if it does not exist
    if not os.path.exists(png_root):
        os.makedirs(png_root, exist_ok=True)

    # iterate over the dicom root directory
    for root, _, files in os.walk(dicom_root):
        for file in tqdm.tqdm(files):
            # check if the file is a dicom file
            if file.endswith(".dicom"):
                dicom_path = os.path.join(root, file)
                png_path = os.path.join(png_root, file.replace(".dicom", ".png"))

                # preprocess the dicom image
                image_preprocessing_pipeline(dicom_path, png_path)


if __name__ == "__main__":
    dicom_root = "/scratch/hekalo/Datasets/vindr/dicom/"
    png_root = "/scratch/hekalo/Datasets/vindr/png/"

    preprocess_dataset(dicom_root, png_root)

