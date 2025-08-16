import os
import glob
import numpy as np

from model.hyperparameters import hyp
from model.training import train_randlanet_model
from model.dataset_npz import RandlanetNpzDataset

# Example: change these to your NPZ dataset directories
# Each directory can either contain a single 'pc.npz' (with xyz, rgb, labels)
# or multiple tile npz files (each containing xyz, rgb, labels)
train_npz_dirs = ["npz_dataset/train/"]
test_npz_dirs = ["npz_dataset/val/"]

# Infer number of classes from available NPZ labels across train dirs
all_labels = set()
for root_dir in train_npz_dirs:
    if os.path.isdir(root_dir):
        npz_files = glob.glob(os.path.join(root_dir, '*.npz'))
        for f in npz_files:
            try:
                with np.load(f) as d:
                    if 'labels' in d:
                        all_labels.update(np.unique(d['labels']).astype(int).tolist())
            except Exception:
                pass

if len(all_labels) > 0:
    hyp['num_classes'] = len(sorted(all_labels))

train_randlanet_model(train_set_list=train_npz_dirs,
                      test_set_list=test_npz_dirs,
                      hyperpars=hyp,
                      use_mlflow=False,
                      num_workers=4,
                      model_name="repo_example_npz",
                      dataset_cls=RandlanetNpzDataset)