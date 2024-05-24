import os
from tqdm import tqdm

import pydicom
import pandas as pd


def vindr_to_yolo_format(annotations, image_width, image_height) -> pd.DataFrame:
    """
    Processes the annotations to a YOLOv5 format txt file.

    :param annotations: The annotations for the given image as a pandas DataFrame.
    :param image_width: The width of the image.
    :param image_height: The height of the image.

    :return: The processed annotations as a pandas DataFrame.
    """

    # get the class id
    class_id = annotations["class_id"].values[0]

    # get the coordinates
    x_min = annotations["x_min"]
    y_min = annotations["y_min"]
    x_max = annotations["x_max"]
    y_max = annotations["y_max"]

    # calculate the yolo coordinates
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = (x_max - x_min)
    height = (y_max - y_min)

    # normalize the coordinates
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    return pd.DataFrame({
        "class_id": [class_id],
        "x_center": [x_center],
        "y_center": [y_center],
        "width": [width],
        "height": [height]
    })


def save_to_txt(annotations, image_id, output_path):
    """
    Saves the annotations to a txt file in YOLOv5 format.

    :param annotations: The annotations as a pandas DataFrame.
    :param image_id: The image id used for the txt file name.
    :param output_path: The path to save the txt file.
    """

    # get the file name
    file_name = os.path.join(output_path, f"{image_id}.txt")

    # save the annotations to a txt file
    annotations.to_csv(file_name, sep=" ", index=False, header=False)


def preprocess_annotations(annotations_file, dicom_path, output_path):
    """
    Preprocesses the annotations file to YOLOv5 format.
    The annotations file is a csv file of all annotations with the following columns:
    image_id, class_name, class_id, (rad_id), x_min, y_min, x_max, y_max
    The coordinates are given in the original image size.
    The YOLOv5 format is a txt file for each image with the following columns:
    class_id, x_center, y_center, width, height
    The coordinates are normalized to the range [0, 1].
    If an image has no annotations (i.e. "No finding"), no txt file is created.

    :param annotations_file: The path to the annotations file.
    :param dicom_path: The path to the dicom images. Images should be located in the root directory as {image_id}.dicom.
    :param output_path: The path to save the processed annotations.
    """

    assert os.path.exists(annotations_file), f"File not found: {annotations_file}"
    assert os.path.exists(dicom_path), f"Directory not found: {dicom_path}"

    # load the annotations file
    annotations = pd.read_csv(annotations_file)

    # create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # remove all rows with "No finding"
    annotations = annotations[annotations["class_name"] != "No finding"]

    # group by image_id
    grouped_data = annotations.groupby("image_id")

    # iterate over the grouped data
    for image_id, annotations in tqdm(grouped_data):
        # get the dicom image
        dicom_image = os.path.join(dicom_path, f"{image_id}.dicom")
        dicom = pydicom.dcmread(dicom_image)
        image_height, image_width = dicom.pixel_array.shape

        # process the annotations
        yolo_annotations = vindr_to_yolo_format(annotations, image_width, image_height)

        # save the annotations to a txt file
        save_to_txt(yolo_annotations, image_id, output_path)


if __name__ == "__main__":
    ann_file = "/scratch/hekalo/Datasets/vindr/annotations.csv"
    dcm_path = "/scratch/hekalo/Datasets/vindr/dicom/"
    annotations_path = "/scratch/hekalo/Datasets/vindr/annotations/"

    preprocess_annotations(ann_file, dcm_path, annotations_path)
