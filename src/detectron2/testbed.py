import json
import os


def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


coco_annotations = load_json("../../data-sample/coco_annotations.json")

for key in coco_annotations.keys():
    print(key)
print(len(coco_annotations["images"]))
print(len(coco_annotations["annotations"]))
