import os
import shutil

from tqdm import tqdm


def create_split(images, train_ratio):
    """
    Creates a split of the images into training, validation, and test sets.
    The split is based on the ratios provided.
    Test ratio is inferred from the train and val ratios.

    :param images: List of all images.
    :param train_ratio: Percentage of training images from the full dataset.
    :return: List of training and validation images.
    """

    # calculate the number of images for each set
    num_images = len(images)
    num_train = int(num_images * train_ratio)
    num_val = num_images - num_train

    # create the split
    train_images = images[:num_train]
    val_images = images[num_train:]

    return train_images, val_images


def create_yolov8_dataset(trainval_png_path, test_png_path, annotations_path, output_path, train_ratio=0.9):
    """
    Creates a dataset split from the png images and annotations.

    :param trainval_png_path: The path to the png images of the train and val sets.
    :param test_png_path: The path to the png images of the test set.
    :param annotations_path: The path to the annotations.
    :param output_path: The path to save the dataset split.
    :param train_ratio: Percentage of training images from the full dataset.
    """

    if not os.path.exists(trainval_png_path):
        raise FileNotFoundError(f"Path not found: {trainval_png_path}")
    if not os.path.exists(test_png_path):
        raise FileNotFoundError(f"Path not found: {test_png_path}")
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Path not found: {annotations_path}")
    if not train_ratio > 0.0 and train_ratio < 1.0:
        raise ValueError(f"Invalid train ratio: {train_ratio}")

    # create the split
    images_trainval = [os.path.join(trainval_png_path, img) for img in os.listdir(trainval_png_path) if img.endswith(".png")]
    image_splits = create_split(images_trainval, train_ratio)
    images_test = [os.path.join(test_png_path, img) for img in os.listdir(test_png_path) if img.endswith(".png")]

    # create the directories
    # yolo requires an 'images' and a 'labels' directory
    # the splits are 'train', 'val', and 'test' within these directories
    directories = ["images", "labels"]
    # split_names = ["train", "val", "test"]
    split_names = ["train", "val"]

    for directory in directories:
        for split in split_names:
            os.makedirs(os.path.join(output_path, directory, split), exist_ok=True)

    # copy the images and annotations to the output directory
    for split, images in zip(split_names, image_splits):
        print(f"Copying {split} images and annotations...")
        for image in tqdm(images):
            image_id = os.path.basename(image).replace(".png", "")
            shutil.copy(image, os.path.join(output_path, "images", str(split), f"{image_id}.png"))
            if os.path.exists(os.path.join(annotations_path, f"{image_id}.txt")):
                shutil.copy(os.path.join(annotations_path, f"{image_id}.txt"),
                            os.path.join(output_path, "labels", str(split), f"{image_id}.txt"))

    print(f"Copying test images and annotations...")
    for image in tqdm(images_test):
        image_id = os.path.basename(image).replace(".png", "")
        shutil.copy(image, os.path.join(output_path, "images", "test", f"{image_id}.png"))
        try:
            shutil.copy(os.path.join(annotations_path, f"{image_id}.txt"),
                        os.path.join(output_path, "labels", "test", f"{image_id}.txt"))
        except FileNotFoundError:
            print(f"No annotations found for {image_id}. Skipping...")


if __name__ == "__main__":
    trainval_root = "/scratch/hekalo/Datasets/vindr/trainval/png/"
    test_root = "/scratch/hekalo/Datasets/vindr/test/png/"
    annotations_root = "/scratch/hekalo/Datasets/vindr/annotations/"
    output_dir = "/scratch/hekalo/Datasets/vindr/dataset/"

    create_yolov8_dataset(trainval_root, test_root, annotations_root, output_dir, train_ratio=0.9)


