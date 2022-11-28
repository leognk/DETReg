import numpy as np
import os
from classes import *
import json

dir = "representer_points/arrays"
data_root = "/home/user/fiftyone/coco-2017"

train_img_ids = np.load(os.path.join(dir, "train_img_ids.npy"))
val_img_ids = np.load(os.path.join(dir, "val_img_ids.npy"))
k = np.load(os.path.join(dir, "img_repr_points.npy"))
with open(os.path.join(data_root, "train/labels.json"), 'r') as f:
    train_labels = json.load(f)
with open(os.path.join(data_root, "validation/labels.json"), 'r') as f:
    val_labels = json.load(f)

print("k has been loaded")

n_train = k.shape[0]
n_val = k.shape[1]
class_id_end = max(ID2CLASS) + 1
k_base = len(BASE_CLASSES)
k_novel = len(NOVEL_CLASSES)

train_img_id2idx = {id: i for i, id in enumerate(train_img_ids)}
val_img_id2idx = {id: i for i, id in enumerate(val_img_ids)}

# Count the number of annotations per class and per training image.
n_train_anns_by_cat = np.zeros((class_id_end, n_train), dtype=np.int32)
n_train_anns = np.zeros(n_train, dtype=np.int32)
for ann in train_labels["annotations"]:
    img_id = ann["image_id"]
    if img_id not in train_img_id2idx:
        continue
    cat_id = ann["category_id"]
    img_idx = train_img_id2idx[img_id]
    n_train_anns_by_cat[cat_id, img_idx] += 1
    n_train_anns[img_idx] += 1

# Count the number of annotations per class and per validation image.
n_val_anns_by_cat = np.zeros((class_id_end, n_val), dtype=np.int32)
n_val_anns = np.zeros(n_val, dtype=np.int32)
for ann in val_labels["annotations"]:
    cat_id = ann["category_id"]
    img_idx = val_img_id2idx[ann["image_id"]]
    n_val_anns_by_cat[cat_id, img_idx] += 1
    n_val_anns[img_idx] += 1

# cat_in_img = (n_val_anns_by_cat[SORTED_NOVEL_CLASSES_IDS] != 0).T
# k_cat = k[:, cat_in_img].flatten()
# k_no_cat = k[:, np.logical_not(cat_in_img)].flatten()
# print("Computing the means")
# print(np.abs(k_cat).mean())
# print(np.abs(k_no_cat).mean())

# Change 0s to 1s to prevent zero division.
n_train_anns[n_train_anns == 0] = 1
n_val_anns[n_val_anns == 0] = 1

r_train = (n_train_anns_by_cat[SORTED_BASE_CLASSES_IDS] / np.expand_dims(n_train_anns, 0))
r_test = (n_val_anns_by_cat[SORTED_NOVEL_CLASSES_IDS] / np.expand_dims(n_val_anns, 0))

print("r has been calculated")

import time
start_time = time.time()
k2 = np.swapaxes(np.tensordot(r_train, k.astype(np.float32), axes=[1, 0]), 1, 2)
etime = time.time() - start_time
print(f"k2 has been calculated (took {etime / 60} min)")

infl = (np.expand_dims(r_test, 0) * k2).sum(-1)

np.save(os.path.join(dir, "influences.npy"), infl)
print("infl has been saved")