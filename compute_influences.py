import numpy as np
import os
from classes import *

dir = "representer_points/tiny_arrays"

train_img_ids = np.load(os.path.join(dir, "train_img_ids.npy"))
val_img_ids = np.load(os.path.join(dir, "val_img_ids.npy"))
k = np.load(os.path.join(dir, "img_repr_points.npy"))

n_train = k.shape[1]
n_test = k.shape[2]
k_base = len(BASE_CLASSES)
k_novel = len(NOVEL_CLASSES)

r_train = np.zeros((k_base, n_train), dtype=np.float16)
r_test = np.zeros((k_novel, n_test), dtype=np.float16)

 