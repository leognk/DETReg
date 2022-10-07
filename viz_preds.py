from collections import defaultdict
import shutil
import os
import json
import copy
import numpy as np
from classes import *

dir_to_remove = "/home/user/.fiftyone"
if os.path.isdir(dir_to_remove):
    shutil.rmtree(dir_to_remove)

import fiftyone as fo
import fiftyone.utils.coco as fouc

class Dataset:

    def __init__(self, data_root, classes_split, n_shots, topk_filtering=None):
        self.data_root = data_root
        self.classes_split = classes_split
        self.n_shots = n_shots
        self.topk_filtering = topk_filtering
        self.load_dataset_with_preds()

    def load_dataset_with_preds(self):

        # The directory containing the source images
        data_path = os.path.join(self.data_root, "validation/data")

        # The path to the COCO labels JSON file
        labels_path = os.path.join(self.data_root, f"validation/labels.json")

        # Import the dataset
        self.dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=data_path,
            labels_path=labels_path,
            label_field="ground_truth",
            # max_samples=100,
        )

        # Add an image_id field to the samples.
        for sample in self.dataset.iter_samples(progress=True):
            id = int(os.path.splitext(os.path.basename(sample["filepath"]))[0])
            sample["image_id"] = id
            sample.save()

        # The path to the predictions file
        preds_path = "exps/DETReg_fs/predictions.json"

        # Load the predictions file
        with open(preds_path, 'r') as f:
            preds = json.load(f)

        # Remove undefined category ids in the predictions and only keep the images in the dataset
        preds_cat_ids = {ann['category_id'] for ann in preds["annotations"]}
        cat_ids = {cat['id'] for cat in preds["categories"]}
        undefined_cat_ids = preds_cat_ids.difference(cat_ids)
        dataset_img_ids = {sample["image_id"] for sample in self.dataset}
        preds["annotations"] = [
            ann for ann in preds["annotations"]\
                if (ann['image_id'] in dataset_img_ids and ann['category_id'] not in undefined_cat_ids)
        ]

        if self.topk_filtering is not None:
            preds = self.filter_topk(preds, self.topk_filtering)

        n_preds = len(preds['annotations'])
        n_gt = self.dataset.count("ground_truth.detections")
        print(f"Found {n_preds:,} predictions ({n_gt:,} ground-truth labels)")

        # Add the predictions to the dataset
        fouc.add_coco_labels(
            self.dataset,
            "predictions",
            preds['annotations'],
            ID2CLASS,
            coco_id_field="image_id",
        )

    def start_app(self, dataset):
        self.session = fo.launch_app(dataset)
    
    def end_app(self):
        self.session.close()

    # Only keep preds with confidence >= confidence_threshold
    def filter_confidence(self, dataset, confidence_threshold):
        return dataset.filter_labels(
            "predictions",
            fo.ViewField("confidence") > confidence_threshold,
            only_matches=False,
        )

    def filter_topk(self, preds, k):

        res = copy.deepcopy(preds)
        res['annotations'] = []

        img_to_anns = defaultdict(list)
        for ann in preds['annotations']:
            img_to_anns[ann['image_id']].append(ann)

        for _, anns in img_to_anns.items():
            topk_ids = np.argsort([ann['score'] for ann in anns])[-k:]
            res['annotations'].extend([anns[i] for i in topk_ids])

        return res

    def compute_performance(self, dataset, classes=None):
        return dataset.evaluate_detections(
            "predictions",
            gt_field="ground_truth",
            eval_key="eval",
            classes=classes,
            compute_mAP=True,
        )

if __name__ == '__main__':

    data_root = "/home/user/fiftyone/coco-2017"
    classes_split = "original"
    n_shots = 30
    topk_filtering = None
    
    d = Dataset(data_root, classes_split, n_shots, topk_filtering)