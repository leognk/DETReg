import numpy as np
import os
from classes import *

dir = "representer_points/tiny_arrays"

train_img_ids = np.load(os.path.join(dir, "train_img_ids.npy"))

print(len(train_img_ids), len(np.unique(train_img_ids)))