import numpy as np
import os
from classes import *
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

dir = "representer_points/arrays"

infl = np.load(os.path.join(dir, "influences.npy"))
infl = infl / np.max(np.abs(infl), axis=0, keepdims=True)
# infl = infl / np.max(np.abs(infl), keepdims=True)
infl = np.round(100 * infl).astype(int)

infl = DataFrame(infl, index=SORTED_BASE_CLASSES, columns=SORTED_NOVEL_CLASSES)
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(infl, linewidths=0, annot=True, fmt=".0f", ax=ax)