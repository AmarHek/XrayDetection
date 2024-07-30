import os
import glob

def check_dataset_integrity(image_dir, label_dir):
    images = glob.glob(os.path.join(image_dir, '*.png'))
    missing_labels = []

    for img_path in images:
        label_path = os.path.join(label_dir, os.path.basename(img_path).replace('.png', '.txt'))
        if not os.path.exists(label_path):
            missing_labels.append(label_path)

    if missing_labels:
        print(f"Missing labels for {len(missing_labels)} images:")
        for lbl in missing_labels:
            print(lbl)
    else:
        print("All images have corresponding labels.")


# Paths to your train images and labels
train_image_dir = '/scratch/hekalo/Datasets/vindr/dataset/images/train'
train_label_dir = '/scratch/hekalo/Datasets/vindr/dataset/labels/train'

check_dataset_integrity(train_image_dir, train_label_dir)
