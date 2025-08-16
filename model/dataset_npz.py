import os
import glob
import json
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
from torch.utils import data

from .utils import rotate


class RandlanetNpzDataset(data.Dataset):

    def __init__(self, pc_path_list, **kwargs):
        self.cfg = kwargs
        self.size = 0

        self.kdtrees = dict()
        self.colors = dict()
        self.labels = dict()
        self.pc_class_count = dict()
        self.total_class_count = defaultdict(int)
        self.total_class_weight = dict()
        self.n_points = 0

        # Load all PCs once; infer label set as union across provided paths
        loaded_pcs = []  # tuples: (pc_id, xyz, rgb, labels, name)
        all_labels = set()

        for ith, pc_dir in enumerate(pc_path_list):
            pc_id, name, xyz, rgb, lbls = self._load_pc_from_dir(pc_dir, fallback_pc_id=ith)
            loaded_pcs.append((pc_id, name, xyz, rgb, lbls))
            all_labels.update(np.unique(lbls).tolist())

        if len(all_labels) == 0:
            raise RuntimeError("No labels found in provided NPZ dataset paths")

        self.mapping = {label: i for i, label in enumerate(sorted(all_labels))}

        for pc_id, name, xyz, rgb, lbls in loaded_pcs:
            kdtree = cKDTree(xyz, leafsize=50)
            self.kdtrees[pc_id] = kdtree
            # Normalize rgb to [0,1] if provided as 0-255
            rgb = rgb.astype(np.float32)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            self.colors[pc_id] = rgb
            self.labels[pc_id] = lbls
            self.size += len(kdtree.data)

            labels_unique, counts = np.unique(lbls, return_counts=True)
            self.pc_class_count[pc_id] = {int(l): int(c) for l, c in zip(labels_unique, counts)}
            for label, count in self.pc_class_count[pc_id].items():
                self.total_class_count[label] += count
                self.n_points += count

        for label, count in self.total_class_count.items():
            self.total_class_weight[label] = count / self.n_points

    def __getitem__(self, _tuple):
        pc_id = _tuple[0]
        pick_point = _tuple[1]
        points = np.array(self.kdtrees[pc_id].data, copy=False)

        # takes the indices of num_points neighbours
        query_idx = self.kdtrees[pc_id].query(pick_point, k=self.cfg['num_points'])[1][0]
        # shuffle index inplace
        rng = np.random.default_rng()
        rng.shuffle(query_idx)

        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[query_idx]

        queried_pc_xyz[:, 0:3] = queried_pc_xyz[:, 0:3] - pick_point[:, 0:3]
        queried_pc_colors = self.colors[pc_id][query_idx]
        queried_pc_labels = self.labels[pc_id][query_idx]

        queried_pc_labels = np.array([self.mapping[lbl] for lbl in queried_pc_labels])

        input_list = self.build_input(queried_pc_xyz, queried_pc_colors, queried_pc_labels, query_idx, pc_id)
        return input_list

    def __len__(self):
        return self.size

    def build_input(self, xyz, rgb, labels, query_idx, pc_id):
        import torch
        from scipy.spatial import cKDTree as _cKDTree

        features = torch.tensor(self.augment_input(xyz, rgb), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        query_idx = torch.tensor(query_idx, dtype=torch.int32)
        pc_id = torch.tensor(pc_id, dtype=torch.int32)
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(self.cfg['num_layers']):
            _, neigh_idx = _cKDTree(xyz).query(xyz, k=self.cfg['k_n'])
            sub_sampling_idx = len(xyz) // self.cfg['sub_sampling_ratio'][i]
            sub_points = xyz[:sub_sampling_idx]
            pool_i = neigh_idx[:sub_sampling_idx]
            _, up_i = _cKDTree(sub_points).query(xyz, k=1)
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
        transformed_xyz = rotate(xyz, [0.0, 0.0, theta])

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
                symmetries.append(np.round(np.random.uniform()) * 2 - 1)
            else:
                symmetries.append(1.0)
        scales *= np.array(symmetries)

        transformed_xyz = transformed_xyz * scales

        noise = np.random.normal(scale=self.cfg['augment_noise'], size=transformed_xyz.shape)
        transformed_xyz = transformed_xyz + noise

        stacked_features = np.concatenate([transformed_xyz, rgb], axis=-1)
        return stacked_features

    def _read_metadata_json(self, pc_dir):
        meta_path = os.path.join(pc_dir, 'metadata', 'metadata.json')
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _infer_pc_id_from_path(self, pc_dir):
        # Try to parse `pc_id=123` pattern; otherwise fallback to hash of name
        base = os.path.basename(os.path.normpath(pc_dir))
        if base.startswith('pc_id='):
            try:
                return int(base.split('=')[1])
            except Exception:
                pass
        # fallback deterministic hash
        return abs(hash(base)) % (10 ** 9)

    def _load_pc_from_dir(self, pc_dir, fallback_pc_id):
        meta = self._read_metadata_json(pc_dir)
        pc_id = (meta.get('pc_id') if meta and isinstance(meta.get('pc_id'), int)
                 else self._infer_pc_id_from_path(pc_dir))
        name = (meta.get('name') if meta and isinstance(meta.get('name'), str)
                else os.path.basename(os.path.normpath(pc_dir)))

        file_pc = os.path.join(pc_dir, 'pc.npz')
        xyz_list = []
        rgb_list = []
        lbl_list = []

        if os.path.isfile(file_pc):
            with np.load(file_pc) as data:
                xyz = data['xyz'].astype(np.float32)
                rgb = data['rgb']
                labels = data['labels']
                labels = labels.reshape(-1)
            xyz_list.append(xyz)
            rgb_list.append(rgb)
            lbl_list.append(labels)
        else:
            # gather all .npz tiles inside the directory (non-recursive)
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
        return pc_id, name, xyz, rgb, labels