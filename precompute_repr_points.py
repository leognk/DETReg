from compute_loss import *
from tqdm import tqdm
import numpy as np
import os
import argparse


class RepresenterPointsPrecomputer:

    def __init__(self, args):
        self.args = args
        self.model, self.criterion, postprocessors = build_model(args)
        self.model.to("cuda:0")

        # Freeze all but the classifier.
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.class_embed[-1].parameters():
            p.requires_grad = True
    
    def precompute(self):
        # Compute the trainset's features and grads (for alpha).
        base_trainset = build_dataset(image_set="train", args=self.args, split="base")
        novel_trainset = build_dataset(image_set="train", args=self.args, split="novel")
        base_train_loader = build_data_loader(self.args, base_trainset)
        novel_train_loader = build_data_loader(self.args, novel_trainset)
        print("Start base trainset.")
        base_features, base_grads, base_img_ids = self.compute_features_and_grads(base_train_loader)
        print("Completed base trainset.")
        print("Start novel trainset.")
        novel_features, novel_grads, novel_img_ids = self.compute_features_and_grads(novel_train_loader)
        # Remove duplicate images
        duplicate_ids = np.array(list(set(base_img_ids) & set(novel_img_ids)))
        novel_img_indices = np.isin(novel_img_ids, duplicate_ids, invert=True)
        novel_features = novel_features[novel_img_indices]
        novel_grads = novel_grads[novel_img_indices]
        novel_img_ids = novel_img_ids[novel_img_indices]
        print("Completed novel trainset.")
        self.train_features = np.concatenate([base_features, novel_features])
        self.grads = np.concatenate([base_grads, novel_grads])
        self.train_img_ids = np.concatenate([base_img_ids, novel_img_ids])

        # Compute the valset's features.
        valset = build_dataset(image_set="val", args=self.args)
        val_loader = build_data_loader(self.args, valset)
        print("Start valset.")
        self.val_features, self.val_img_ids = self.compute_features(val_loader)
        print("Completed valset.")

        self.save()
    
    def save(self):
        dir = os.path.join(self.args.repr_dir, "arrays")
        os.makedirs(dir, exist_ok=True)
        np.save(os.path.join(dir, "train_features.npy"), self.train_features)
        np.save(os.path.join(dir, "grads.npy"), self.grads)
        np.save(os.path.join(dir, "train_img_ids.npy"), self.train_img_ids)

        np.save(os.path.join(dir, "val_features.npy"), self.val_features)
        np.save(os.path.join(dir, "val_img_ids.npy"), self.val_img_ids)

    def to_numpy(self, t):
        return t.detach().cpu().numpy().astype(np.float16)

    def compute_features_and_grads(self, train_loader):
        features = []
        grads = []
        img_ids = []

        for samples, targets in tqdm(train_loader):
            samples, targets = samples.to("cuda:0"), [{k: v.to("cuda:0") for k, v in t.items()} for t in targets]
            outputs = self.model(samples)
            features.append(self.to_numpy(outputs["dec_outputs"][-1]))

            losses = compute_loss(self.criterion, outputs, targets)
            self.model.zero_grad()
            losses.backward()
            grads.append(self.to_numpy(outputs["all_pred_logits"].grad[-1]))

            img_ids += [t["image_id"].item() for t in targets]
        
        return np.concatenate(features), np.concatenate(grads), np.array(img_ids)

    def compute_features(self, test_loader):
        features = []
        img_ids = []

        for samples, targets in tqdm(test_loader):
            samples, targets = samples.to("cuda:0"), [{k: v.to("cuda:0") for k, v in t.items()} for t in targets]
            outputs = self.model(samples)
            features.append(self.to_numpy(outputs["dec_outputs"][-1]))

            img_ids += [t["image_id"].item() for t in targets]
        
        return np.concatenate(features), np.array(img_ids)


if __name__ == '__main__':

    from main import parse_args, set_dataset_path
    from util.default_args import set_model_defaults, get_args_parser

    ###############################################################################################################################################################
    # args = f'''
    # --data_root /home/user/fiftyone
    # --dataset coco
    # --pretrain exps/DETReg_fs/checkpoint.pth
    # --num_workers 4
    # --batch_size 4
    # --repr_dir representer_points
    # '''
    # args = args.split()
    # args = parse_args(args)
    ###############################################################################################################################################################

    ###############################################################################################################################################################
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    set_dataset_path(args)
    ###############################################################################################################################################################

    set_model_defaults(args)
    rpPrecomputer = RepresenterPointsPrecomputer(args)
    rpPrecomputer.precompute()