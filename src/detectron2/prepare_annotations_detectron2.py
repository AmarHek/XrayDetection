import os
from typing import List
import json
from tqdm import tqdm

import pydicom
import pandas as pd


def vindr_to_coco_format(image_id, annotations, old_shape, new_shape) -> List:
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

    # rescale the coordinates to the new image size
    x_min = x_min * new_shape[1] / old_shape[1]
    y_min = y_min * new_shape[0] / old_shape[0]
    x_max = x_max * new_shape[1] / old_shape[1]
    y_max = y_max * new_shape[0] / old_shape[0]

    # calculate the width and height
    width = (x_max - x_min)
    height = (y_max - y_min)

    # iterate the dataframe and create the coco format list of annotations
    coco_annotations = []
    for i in range(len(annotations)):
        coco_annotations.append({
            "id": i,
            "image_id": image_id,
            "category_id": class_id,
            "bbox": [x_min[i], y_max[i], width[i], height[i]]
        })


def preprocess_annotations_coco(dataset_path, categories):
    """
    Preprocesses the annotations file to COCO format.
    The annotations file is a csv file of all annotations with the following columns:
    image_id, class_name, class_id, (rad_id), x_min, y_min, x_max, y_max
    The coordinates are given in the original image size.
    For COCO format, the coordinates are adjusted to fit the resized images.

    :param dataset_path: The path to the full dataset. Requires the following dirs and files:
    dicom: The directory containing all dicom images.
    png: The directory containing all resized png images.
    annotations: The file containing the original annotations.
    Images should be located inside these directories as {image_id}.dicom/png.
    """

    dicom_path = os.path.join(dataset_path, "dicom")
    png_path = os.path.join(dataset_path, "png")
    annotations_path = os.path.join(dataset_path, "annotations.csv")

    assert os.path.exists(dataset_path), f"Directory not found: {dataset_path}"
    assert os.path.exists(dicom_path), f"Directory not found: {dicom_path}"
    assert os.path.exists(png_path), f"Directory not found: {png_path}"
    assert os.path.exists(annotations_path), f"Directory not found: {annotations_path}"

    # load the annotations file
    annotations = pd.read_csv(annotations_path)

    # remove all rows with "No finding"
    annotations = annotations[annotations["class_name"] != "No finding"]

    # group by image_id
    grouped_data = annotations.groupby("image_id")

    # set up the coco format dictionary
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    running_image_id = 1
    # iterate over the grouped data
    for image_id, annotations in tqdm(grouped_data):
        # get the dicom image
        dicom_image = os.path.join(dicom_path, f"{image_id}.dicom")
        old_shape = pydicom.dcmread(dicom_image).pixel_array.shape
        png_image = os.path.join(png_path, f"{image_id}.png")
        new_shape = pydicom.dcmread(png_image).pixel_array.shape

        # process the annotations
        coco_format_annotations = vindr_to_coco_format(image_id, annotations, old_shape, new_shape)

        # add the image to the coco format
        coco_annotations["images"].append({
            "id": running_image_id,
            "file_name": f"{image_id}.png",
            "height": new_shape[0],
            "width": new_shape[1]
        })

        # add the annotations to the coco format
        coco_annotations["annotations"].extend(coco_format_annotations)
        running_image_id += 1

    # save the coco format annotations to a json file

    with open(os.path.join(dataset_path, "coco_annotations.json"), "w") as f:
        json.dump(coco_annotations, f)


if __name__ == "__main__":
    categories = [
        {"id": 0, "name": "Aortic enlargement"},
        {"id": 1, "name": "Atelectasis"},
        {"id": 2, "name": "Calcification"},
        {"id": 3, "name": "Cardiomegaly"},
        {"id": 4, "name": "Consolidation"},
        {"id": 5, "name": "ILD"},
        {"id": 6, "name": "Infiltration"},
        {"id": 7, "name": "Lung Opacity"},
        {"id": 8, "name": "Nodule/Mass"},
        {"id": 9, "name": "Other lesion"},
        {"id": 10, "name": "Pleural effusion"},
        {"id": 11, "name": "Pleural thickening"},
        {"id": 12, "name": "Pneumothorax"},
        {"id": 13, "name": "Pulmonary fibrosis"}
    ]

    dataset_path = "/scratch/hekalo/Datasets/vindr/"
    preprocess_annotations_coco(dataset_path, categories)
