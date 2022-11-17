import json
import os
from classes import NOVEL_CLASSES_IDS

data_root = "/home/user/fiftyone/coco-2017"
fs_labels_path = os.path.join(data_root, "train", "subsets", "few_shots_labels_200shots", "seed0.json")
with open(fs_labels_path, 'r') as f:
    labels = json.load(f)

cats = []
for cat in labels['categories']:
    if cat['id'] in NOVEL_CLASSES_IDS:
        cats.append(cat)

anns = []
img_ids = set()
for ann in labels['annotations']:
    if ann['category_id'] in NOVEL_CLASSES_IDS:
        anns.append(ann)
        img_ids.add(ann['image_id'])

imgs = []
for img in labels['images']:
    if img['id'] in img_ids:
        imgs.append(img)

novel_labels = {'categories': cats, 'annotations': anns, 'images': imgs}

dir = os.path.join(data_root, "train", "subsets", "novel_few_shots_labels_200shots")
os.makedirs(dir, exist_ok=True)
with open(os.path.join(dir, "seed0.json"), 'w') as f:
    json.dump(novel_labels, f)