import shutil
import os
import json
from classes import *

dir_to_remove = "/home/user/.fiftyone"
if os.path.isdir(dir_to_remove):
    shutil.rmtree(dir_to_remove)

import fiftyone as fo
import fiftyone.utils.coco as fouc

class Dataset:

    def __init__(self, data_path, labels_path, preds_path):
        self.data_path = data_path
        self.labels_path = labels_path
        self.preds_path = preds_path
        self.load_dataset_with_preds()

    def load_dataset_with_preds(self):

        # Import the dataset
        self.dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=self.data_path,
            labels_path=self.labels_path,
            label_field="ground_truth",
            # max_samples=100,
        )

        # Add an image_id field to the samples.
        for sample in self.dataset.iter_samples(progress=True):
            id = int(os.path.splitext(os.path.basename(sample["filepath"]))[0])
            sample["image_id"] = id
            sample.save()

        # Load the predictions file
        with open(self.preds_path, 'r') as f:
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
    data_path = os.path.join(data_root, "validation/data")
    labels_path = os.path.join(data_root, "validation/subsets/original_base_classes_labels.json")
    predictions_path = "exps/DETReg_fine_tune_base_classes_original/predictions.json"
    
    d = Dataset(data_path, labels_path, predictions_path)