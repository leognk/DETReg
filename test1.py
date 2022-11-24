import numpy as np
import os
from classes import *

dir = "representer_points/arrays"

train_ann_ids = np.load(os.path.join(dir, "train_ann_ids.npy"))
val_ann_ids = np.load(os.path.join(dir, "val_ann_ids.npy"))
grads = np.load(os.path.join(dir, "grads.npy"))
train_features = np.load(os.path.join(dir, "train_features.npy"))
val_features = np.load(os.path.join(dir, "val_features.npy"))

lmbd = 1e-4
alphas = grads / (-2 * lmbd * len(grads))

# for i in range(len(train_features)):
#     for t in range(len(val_features)):
#         sims = np.matmul(train_features[i], val_features[t].T)
#         for cb in BASE_CLASSES_IDS:
#             for cn in NOVEL_CLASSES_IDS:
#                 k_cn = alphas[i, :, cn].reshape((-1, 1)) * sims