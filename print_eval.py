import numpy as np
import torch
from collections import defaultdict
from classes import *

class EvaluationLoader:

    def __init__(self, eval_path):
        self.eval = torch.load(eval_path)
        self.compute_APs()

    def compute_APs(self):
        area_range = 'all'
        max_detections = 100
        dict_class_ids = {'All': None}
        dict_class_ids['Novel'] = {CLASS2ID[c] for c in NOVEL_CLASSES}
        dict_class_ids['Base'] = {CLASS2ID[c] for c in BASE_CLASSES}
        iou_thresholds = {'AP': None, 'AP_50': 0.5, 'AP_75': 0.75}

        self.scores = defaultdict(dict)
        p = self.eval['params']

        area_range_idx = [
            i for i, aRng in enumerate(p.areaRngLbl) if aRng == area_range
        ]
        max_detections_idx = [
            i for i, mDet in enumerate(p.maxDets) if mDet == max_detections
        ]

        for class_split, class_ids in dict_class_ids.items():
            scores = self.eval['precision'].copy()
            if class_ids is not None:
                category_ids = [
                    i for i, catId in enumerate(p.catIds) if catId in class_ids
                ]
                scores = scores[:, :, category_ids]
            for iou_threshold_str, iou_threshold in iou_thresholds.items():
                s = scores.copy()
                if iou_threshold is not None:
                    t = np.where(p.iouThrs == iou_threshold)[0]
                    s = s[t]
                s = s[:, :, :, area_range_idx, max_detections_idx]
                avg_score = 100 * np.mean(s[s > -1])
                self.scores[class_split][iou_threshold_str] = avg_score
        
    def __repr__(self):
        res = ""
        for class_split in self.scores:
            row = ""
            for iouThrs, score in self.scores[class_split].items():
                if row != "": row += " | "
                class_split_format = "" if class_split == "All" else f"{class_split} "
                row += f"{class_split_format}{iouThrs}: {round(score, 1)}"
            if res != "": res += "\n"
            res += row
        return res

if __name__ == '__main__':

    eval = EvaluationLoader("exps/DETReg_fs/eval5.pth")
    print(eval)