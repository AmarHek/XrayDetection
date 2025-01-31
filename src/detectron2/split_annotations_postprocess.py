import json
import os
import shutil


def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


def save_json(data, json_file):
    with open(json_file, 'w') as f:
        json.dump(data, f)


def split_annotations(annotation_file, train_dir, val_dir, output_dir):
    coco_data = load_json(annotation_file)

    # Create dictionaries to hold new data
    train_data = {k: [] for k in coco_data if isinstance(coco_data[k], list)}
    val_data = {k: [] for k in coco_data if isinstance(coco_data[k], list)}

    train_img_ids = set()
    val_img_ids = set()

    def get_image_ids(directory, img_ids):
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                for img in coco_data['images']:
                    if img['file_name'] == filename:
                        img_ids.add(img['id'])
                        break

    get_image_ids(train_dir, train_img_ids)
    get_image_ids(val_dir, val_img_ids)

    def filter_data(img_ids, target_data):
        for img in coco_data['images']:
            if img['id'] in img_ids:
                target_data['images'].append(img)

        for ann in coco_data['annotations']:
            if ann['image_id'] in img_ids:
                target_data['annotations'].append(ann)

        target_data['categories'] = coco_data['categories']
        if 'licenses' in coco_data:
            target_data['licenses'] = coco_data['licenses']
        if 'info' in coco_data:
            target_data['info'] = coco_data['info']

    filter_data(train_img_ids, train_data)
    filter_data(val_img_ids, val_data)

    os.makedirs(output_dir, exist_ok=True)
    save_json(train_data, os.path.join(output_dir, 'instances_train.json'))
    save_json(val_data, os.path.join(output_dir, 'instances_val.json'))

    print(f"Saved {len(train_data['images'])} train images and {len(train_data['annotations'])} annotations.")
    print(f"Saved {len(val_data['images'])} val images and {len(val_data['annotations'])} annotations.")


if __name__ == "__main__":
    root = "/scratch/hekalo/Datasets/vindr/"
    annotations_file = os.path.join(root, "coco_annotations_trainval.json")
    output_dir = os.path.join(root, "dataset", "annotations")
    image_dir = os.path.join(root, "dataset", "images")
    train_dir = os.path.join(image_dir, "train")
    val_dir = os.path.join(image_dir, "val")

    split_annotations(annotations_file, train_dir, val_dir, output_dir)
