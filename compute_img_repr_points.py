import numpy as np
import os
from classes import *
from tqdm import tqdm
import torch

dir = "representer_points/arrays"

grads = np.load(os.path.join(dir, "grads.npy"))
f_train = np.load(os.path.join(dir, "train_features.npy"))
f_test = np.load(os.path.join(dir, "val_features.npy"))

n_train = len(f_train)
n_test = len(f_test)
k_novel = len(NOVEL_CLASSES)
hidden_dim = f_train.shape[-1]
lmbd = 1e-4
device = torch.device("cuda:0")

novel_ids = sorted(list(NOVEL_CLASSES_IDS))
alpha = grads[:, :, novel_ids] / (-2 * lmbd * n_train)
del grads

f_test_sum = f_test.sum(1).T
f_test_sum = torch.from_numpy(f_test_sum).to(device)

k = torch.zeros((k_novel, n_train, n_test), dtype=torch.float16, device=device)
for n in tqdm(range(k_novel)):
    weighted_f_train_sum = (alpha[:, :, n, np.newaxis] * f_train).sum(1)
    weighted_f_train_sum = torch.from_numpy(weighted_f_train_sum).to(device)
    k[n] = torch.matmul(weighted_f_train_sum, f_test_sum)

np.save(os.path.join(dir, "img_repr_points.npy"), k.cpu().numpy())