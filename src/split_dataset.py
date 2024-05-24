import os
import shutil


def create_split(images, train_ratio, val_ratio):
    """
    Creates a split of the images into training, validation, and test sets.
    The split is based on the ratios provided.
    Test ratio is inferred from the train and val ratios.

    :param images: List of all images.
    :param train_ratio: The ratio of the training set.
    :param val_ratio: The ratio of the validation set.
    :return: List of training, validation, and test images.
    """

    assert train_ratio + val_ratio < 1, "The sum of the train and val ratios should be less than 1."

    # calculate the number of images for each set
    num_images = len(images)
    num_train = int(num_images * train_ratio)
    num_val = int(num_images * val_ratio)
    num_test = num_images - num_train - num_val

    # create the split
    train_images = images[:num_train]
    val_images = images[num_train:num_train + num_val]
    test_images = images[num_train + num_val:]

    return (train_images, val_images, test_images)


def create_yolov8_dataset(png_path, annotations_path, output_path, train_ratio=0.7, val_ratio=0.15):
    """
    Creates a dataset split from the png images and annotations.

    :param png_path: The path to the png images.
    :param annotations_path: The path to the annotations.
    :param output_path: The path to save the dataset split.
    :param train_ratio: The ratio of the training set.
    :param val_ratio: The ratio of the validation set.
    """

    assert os.path.exists(png_path), "The png path does not exist."
    assert os.path.exists(annotations_path), "The annotations path does not exist."
    assert 0 < train_ratio < 1, "The train ratio should be between 0 and 1."

    # create the split
    images = [os.path.join(png_path, img) for img in os.listdir(png_path) if img.endswith(".png")]
    image_splits = create_split(images, train_ratio, val_ratio)

    # create the directories
    # yolo requires an 'images' and a 'labels' directory
    # the splits are 'train', 'val', and 'test' within these directories
    directories = ["images", "labels"]
    split_names = ["train", "val", "test"]

    for directory in directories:
        for split in split_names:
            os.makedirs(os.path.join(output_path, directory, split), exist_ok=True)

    # copy the images and annotations to the output directory
    for split, images in zip(split_names, image_splits):
        for image in images:
            image_id = os.path.basename(image).replace(".png", "")
            shutil.copy(image, os.path.join(output_path, "images", str(split), f"{image_id}.png"))
            shutil.copy(os.path.join(annotations_path, f"{image_id}.txt"),
                        os.path.join(output_path, "labels", str(split), f"{image_id}.txt"))


if __name__ == "__main__":
    png_root = "/scratch/hekalo/Datasets/vindr/png/"
    annotations_root = "/scratch/hekalo/Datasets/vindr/annotations/"
    output_dir = "/scratch/hekalo/Datasets/vindr/dataset/"

    create_yolov8_dataset(png_root, annotations_root, output_dir, train_ratio=0.7, val_ratio=0.15)
