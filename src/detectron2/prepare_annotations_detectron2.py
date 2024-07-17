import os
from typing import List
import json
from tqdm import tqdm

import pydicom
import pandas as pd
import cv2


def vindr_to_coco_format(image_id, class_config, annotations, old_shape, new_shape) -> List:
    """
    Processes the annotations to a YOLOv5 format txt file.

    :param annotations: The annotations for the given image as a pandas DataFrame.
    :param old_shape: The shape of the original dicom image.
    :param new_shape: The shape of the processed png image.

    :return: The processed COCO annotations as a list.
    """

    # get the class names and coordinates
    class_names = annotations["class_name"].values
    if "No finding" in class_names:
        return None
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
            "category_id": int(class_config[class_names[i]]),
            "bbox": [x_min.iloc[i], y_max.iloc[i], width.iloc[i], height.iloc[i]]
        })

    return coco_annotations


def preprocess_annotations_coco(image_path, annotations_file, categories, output_file="coco_annotations.csv"):
    """
    Preprocesses the annotations file to COCO format.
    The annotations file is a csv file of all annotations with the following columns:
    image_id, class_name, class_id, (rad_id), x_min, y_min, x_max, y_max
    The coordinates are given in the original image size.
    For COCO format, the coordinates are adjusted to fit the resized images.

    :param image_path: The path to the full dataset. Requires the following dirs and files:
    dicom: The directory containing all dicom images.
    png: The directory containing all resized png images.
    annotations_file: The file containing the original annotations.
    categories: The list of categories in COCO format.
    output_file: The name of the output file.
    Images should be located inside these directories as {image_id}.dicom/png.
    """

    dicom_path = os.path.join(image_path, "dicom")
    png_path = os.path.join(image_path, "png")

    annotations = pd.read_csv(annotations_file)
    grouped_data = annotations.groupby("image_id")

    # set up the coco format dictionary
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    # turn categories into a dictionary of class name to class id
    class_config = {category["name"]: category["id"] for category in categories}

    running_image_id = 1
    for image_id, annotations in tqdm(grouped_data):
        # get the original and new shape of the image
        dicom_image = os.path.join(dicom_path, f"{image_id}.dicom")
        old_shape = pydicom.dcmread(dicom_image).pixel_array.shape
        png_image = os.path.join(png_path, f"{image_id}.png")
        new_shape = cv2.imread(png_image).shape

        # add the image to the coco format
        coco_annotations["images"].append({
            "id": running_image_id,
            "file_name": f"{image_id}.png",
            "height": int(new_shape[0]),
            "width": int(new_shape[1])
        })

        # Only add annotations, if there are boxes, i.e. if there is no "No Finding" present
        # process the annotations
        coco_format_annotations = vindr_to_coco_format(running_image_id, class_config,
                                                       annotations, old_shape, new_shape)
        # add the annotations to the coco format
        if coco_format_annotations:
            coco_annotations["annotations"].extend(coco_format_annotations)
        running_image_id += 1

    # save the coco format annotations to a json file
    with open(os.path.join(output_file), "w") as f:
        json.dump(coco_annotations, f)


if __name__ == "__main__":
    categories = [
        {"id": 0, "name": "Aortic enlargement"},
        {"id": 1, "name": "Atelectasis"},
        {"id": 2, "name": "Calcification"},
        {"id": 3, "name": "Cardiomegaly"},
        {"id": 4, "name": "Clavicle fracture"},
        {"id": 5, "name": "Consolidation"},
        {"id": 6, "name": "Edema"},
        {"id": 7, "name": "Emphysema"},
        {"id": 8, "name": "Enlarged PA"},
        {"id": 9, "name": "ILD"},
        {"id": 10, "name": "Infiltration"},
        {"id": 11, "name": "Lung Opacity"},
        {"id": 12, "name": "Lung cavity"},
        {"id": 13, "name": "Lung cyst"},
        {"id": 14, "name": "Mediastinal shift"},
        {"id": 15, "name": "Nodule/Mass"},
        {"id": 16, "name": "Other lesion"},
        {"id": 17, "name": "Pleural effusion"},
        {"id": 18, "name": "Pleural thickening"},
        {"id": 19, "name": "Pneumothorax"},
        {"id": 20, "name": "Pulmonary fibrosis"},
        {"id": 21, "name": "Rib fracture"}
    ]

    trainval_path = "/scratch/hekalo/Datasets/vindr/trainval/"
    trainval_annotations = "/scratch/hekalo/Datasets/vindr/annotations_train.csv"
    output_path_trainval = "/scratch/hekalo/Datasets/vindr/coco_annotations_trainval.json"
    test_path = "/scratch/hekalo/Datasets/vindr/test/"
    test_annotations = "/scratch/hekalo/Datasets/vindr/annotations_test.csv"
    output_path_test = "/scratch/hekalo/Datasets/vindr/coco_annotations_test.json"

    preprocess_annotations_coco(trainval_path, trainval_annotations,
                                categories, output_file=output_path_trainval)
    preprocess_annotations_coco(test_path, test_annotations,
                                categories, output_file=output_path_test)
