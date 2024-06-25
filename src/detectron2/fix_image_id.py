import json


def correct_image_ids(annotation_file, output_file):
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Create a mapping from filename to the correct image ID
    filename_to_id = {}
    for img in coco_data['images']:
        filename_without_ext = img['file_name'].rsplit('.', 1)[0]
        filename_to_id[filename_without_ext] = img['id']

    # Correct the "image_id" attribute in annotations
    for ann in coco_data['annotations']:
        filename_without_ext = ann['image_id']
        if filename_without_ext in filename_to_id:
            ann['image_id'] = filename_to_id[filename_without_ext]
        else:
            print(f"Warning: No matching image found for annotation with image_id {filename_without_ext}")

    # Save the corrected annotations
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)

    print(f"Corrected image IDs and saved to {output_file}")


# Paths to your data
annotation_file = '../../data-sample/coco_annotations.json'
output_file = '../../data-sample/corrected_annotations.json'

correct_image_ids(annotation_file, output_file)
