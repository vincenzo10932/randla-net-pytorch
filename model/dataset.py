import os.path
import random
from collections import defaultdict
import glob
import json

import torch
import numpy as np
from scipy.spatial import cKDTree

from torch.utils import data

from .utils import rotate


class RandlanetDataset(data.Dataset):

    def __init__(self, pc_path_list, **kwargs):
        self.cfg = kwargs
        self.size = 0

        def _read_metadata_json(pc_dir):
            meta_path = os.path.join(pc_dir, 'metadata', 'metadata.json')
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        return json.load(f)
                except Exception:
                    return None
            return None

        def _infer_pc_id_from_path(pc_dir):
            base = os.path.basename(os.path.normpath(pc_dir))
            if base.startswith('pc_id='):
                try:
                    return int(base.split('=')[1])
                except Exception:
                    pass
            return abs(hash(base)) % (10 ** 9)

        def _load_npz_from_dir(pc_dir):
            file_pc = os.path.join(pc_dir, 'pc.npz')
            xyz_list, rgb_list, lbl_list = [], [], []
            if os.path.isfile(file_pc):
                with np.load(file_pc) as data:
                    xyz = data['xyz'].astype(np.float32)
                    rgb = data['rgb']
                    labels = data['labels'].reshape(-1)
                xyz_list.append(xyz)
                rgb_list.append(rgb)
                lbl_list.append(labels)
            else:
                tile_files = sorted(glob.glob(os.path.join(pc_dir, '*.npz')))
                if len(tile_files) == 0:
                    raise FileNotFoundError(f"No NPZ files found in {pc_dir}")
                for tf in tile_files:
                    with np.load(tf) as data:
                        if not {'xyz', 'rgb', 'labels'}.issubset(set(data.files)):
                            continue
                        xyz_list.append(data['xyz'].astype(np.float32))
                        rgb_list.append(data['rgb'])
                        lbl_list.append(data['labels'].reshape(-1))
            xyz = np.concatenate(xyz_list, axis=0)
            rgb = np.concatenate(rgb_list, axis=0)
            labels = np.concatenate(lbl_list, axis=0)
            return xyz, rgb, labels

        # Determine base label set from first dataset folder
        first_meta = _read_metadata_json(pc_path_list[0])
        if first_meta and 'labels' in first_meta:
            pc_labels = first_meta['labels']
        else:
            _, _, first_labels = _load_npz_from_dir(pc_path_list[0])
            pc_labels = sorted(list(set(first_labels.astype(int).tolist())))

        # Test mode flag
        self.test = pc_labels == [-99.] or pc_labels == [-99]
        if self.test:
            assert len(pc_path_list) == 1, "Only one pc can be used as test"
            pc_labels = [-99.]
        else:
            print("Using labels from first dataset provided")
            assert len(pc_labels) == self.cfg['num_classes'], \
                f"self.cfg['num_classes'] {self.cfg['num_classes']} is different" \
                f"from len(pc_labels) {len(pc_labels)}"
            for pc_path in pc_path_list:
                other_meta = _read_metadata_json(pc_path)
                if other_meta and 'labels' in other_meta:
                    o_pc_labels = other_meta['labels']
                else:
                    _, _, labels_other = _load_npz_from_dir(pc_path)
                    o_pc_labels = sorted(list(set(labels_other.astype(int).tolist())))
                assert set(o_pc_labels).issubset(set(pc_labels)), \
                    "Point clouds must be created considering a subset " \
                    "of labels from the first pc provided"

        self.mapping = {label: i for i, label in enumerate(sorted(pc_labels))}
        self.kdtrees = dict()
        self.colors = dict()
        self.labels = dict()
        self.pc_class_count = dict()
        self.total_class_count = defaultdict(int)
        self.total_class_weight = dict()
        self.n_points = 0

        for pc_path in pc_path_list:
            meta = _read_metadata_json(pc_path)
            pc_id = (meta.get('pc_id') if meta and isinstance(meta.get('pc_id'), int)
                     else _infer_pc_id_from_path(pc_path))
            pc_name = (meta.get('name') if meta and isinstance(meta.get('name'), str)
                       else os.path.basename(os.path.normpath(pc_path)))

            xyz, rgb, labels = _load_npz_from_dir(pc_path)
            print(f"KDtree for pc {pc_id} {pc_name} not found, creating it")
            kdtree = cKDTree(xyz, leafsize=50)
            self.kdtrees[pc_id] = kdtree

            rgb = rgb.astype(np.float32)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            self.colors[pc_id] = rgb
            self.labels[pc_id] = labels
            self.size += len(self.kdtrees[pc_id].data)

            labels_u, counters = np.unique(self.labels[pc_id], return_counts=True)
            self.pc_class_count[pc_id] = dict()
            for label, counter in zip(labels_u, counters):
                self.pc_class_count[pc_id][label] = counter
                self.total_class_count[label] += counter
                self.n_points += counter

        for label, counter in self.total_class_count.items():
            self.total_class_weight[label] = counter / self.n_points

    def __getitem__(self, _tuple):
        pc_id = _tuple[0]
        pick_point = _tuple[1]
        # center_point = _tuple[1].reshape(1, -1)
        # Get all points within the cloud from tree structure
        points = np.array(self.kdtrees[pc_id].data, copy=False)

        query_idx = self.kdtrees[pc_id].query(pick_point,
                                              k=self.cfg['num_points'])[1][0]
        # shuffle index inplace
        rng = np.random.default_rng()
        rng.shuffle(query_idx)

        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[query_idx]

        queried_pc_xyz[:, 0:3] = queried_pc_xyz[:, 0:3] - pick_point[:, 0:3]
        queried_pc_colors = self.colors[pc_id][query_idx]
        queried_pc_labels = self.labels[pc_id][query_idx]

        queried_pc_labels = np.array(
            [self.mapping[lbl] for lbl in queried_pc_labels])

        input_list = self.build_input(queried_pc_xyz, queried_pc_colors,
                                      queried_pc_labels, query_idx, pc_id)

        return input_list

    def __len__(self):
        return self.size

    def build_input(self, xyz, rgb, labels, query_idx, pc_id):
        features = torch.tensor(self.augment_input(xyz, rgb), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        query_idx = torch.tensor(query_idx, dtype=torch.int32)
        pc_id = torch.tensor(pc_id, dtype=torch.int32)
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(self.cfg['num_layers']):
            _, neigh_idx = cKDTree(xyz).query(xyz, k=self.cfg['k_n'])
            sub_sampling_idx = len(xyz)//self.cfg['sub_sampling_ratio'][i]
            sub_points = xyz[:sub_sampling_idx]
            pool_i = neigh_idx[:sub_sampling_idx]
            _, up_i = cKDTree(sub_points).query(xyz, k=1)
            input_points.append(torch.tensor(xyz, dtype=torch.float32))
            input_neighbors.append(torch.tensor(neigh_idx, dtype=torch.int32))
            input_pools.append(torch.tensor(pool_i, dtype=torch.int32))
            input_up_samples.append(torch.tensor(up_i, dtype=torch.int32))
            xyz = sub_points

        inputs = input_points + input_neighbors + input_pools + input_up_samples
        inputs += [features, labels, query_idx, pc_id]
        return inputs

    def augment_input(self, xyz, rgb):
        theta = np.random.uniform(0.0, 2 * np.pi)
        transformed_xyz = rotate(xyz, [0., 0., theta])

        # Choose random scales for each example
        min_s = self.cfg['augment_scale_min']
        max_s = self.cfg['augment_scale_max']
        if self.cfg['augment_scale_anisotropic']:
            scales = np.random.uniform(min_s, max_s, size=(3,))
        else:
            scales = np.random.uniform(min_s, max_s)
            scales = np.array([scales, scales, scales])

        symmetries = []
        for i in range(3):
            if self.cfg['augment_symmetries'][i]:
                symmetries.append(np.round(
                    np.random.uniform()) * 2 - 1)
            else:
                symmetries.append(1.)
        scales *= np.array(symmetries)

        # Apply scales
        transformed_xyz = transformed_xyz * scales

        noise = np.random.normal(scale=self.cfg['augment_noise'],
                                 size=transformed_xyz.shape)
        transformed_xyz = transformed_xyz + noise

        stacked_features = np.concatenate([transformed_xyz, rgb], axis=-1)
        return stacked_features
