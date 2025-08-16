import os
import glob
import json
import numpy as np
import open3d as o3d

# SETTINGS
DATA_DIR = "ply_dataset"        # folder with your .ply files
OUT_DIR = "npz_dataset"         # output folder for .npz tiles
VOXEL_SIZE = 0.04               # meters (4cm typical for indoor scans)
NPTS = 12000                    # points per tile
TILE_SIZE = 2.0                 # meters (tile cube edge length)
OVERLAP = 0.5                   # tile overlap ratio


def voxel_downsample(pcd, voxel_size):
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def crop_tiles(points, colors, labels, tile_size, overlap, npts):
    """
    Split a large scene into fixed-size overlapping tiles
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    stride = tile_size * (1 - overlap)
    tiles = []

    x_steps = np.arange(mins[0], maxs[0], stride)
    y_steps = np.arange(mins[1], maxs[1], stride)

    for x in x_steps:
        for y in y_steps:
            mask = (
                (points[:, 0] >= x) & (points[:, 0] < x + tile_size) &
                (points[:, 1] >= y) & (points[:, 1] < y + tile_size)
            )
            if mask.sum() < 100:  # skip empty tiles
                continue
            pts_tile = points[mask]
            rgb_tile = colors[mask]
            lbl_tile = labels[mask]

            # If > npts, random sample
            if pts_tile.shape[0] > npts:
                idx = np.random.choice(pts_tile.shape[0], npts, replace=False)
            else:
                idx = np.random.choice(pts_tile.shape[0], npts, replace=True)

            tiles.append((pts_tile[idx], rgb_tile[idx], lbl_tile[idx]))
    return tiles


def process_ply(ply_path, out_root):
    fname = os.path.splitext(os.path.basename(ply_path))[0]
    scene_dir = os.path.join(out_root, f"pc_id={fname}")
    os.makedirs(scene_dir, exist_ok=True)
    print(f"Processing {fname} → {scene_dir}")

    # Load ply
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    # Load semantic labels from ply (expects attribute 'sem')
    pcd_t = o3d.t.io.read_point_cloud(ply_path)  # tensor-based to access attributes
    if 'sem' not in pcd_t.point:
        raise KeyError("Ply is missing 'sem' point attribute for labels")
    labels = np.array(pcd_t.point["sem"].numpy(), dtype=np.int16)

    # Downsample
    pcd = voxel_downsample(pcd, VOXEL_SIZE)
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    labels = labels[: points.shape[0]]  # align after voxel downsample

    # Split into tiles
    tiles = crop_tiles(points, colors, labels, TILE_SIZE, OVERLAP, NPTS)

    # Save each tile under scene dir
    for i, (xyz, rgb, lbl) in enumerate(tiles):
        out_path = os.path.join(scene_dir, f"{fname}_tile{i:04d}.npz")
        np.savez_compressed(out_path, xyz=xyz, rgb=rgb, labels=lbl)

    # Write minimal metadata JSON (for NPZ dataset loader)
    meta_dir = os.path.join(scene_dir, 'metadata')
    os.makedirs(meta_dir, exist_ok=True)
    unique_labels = np.unique(labels).astype(int).tolist()
    metadata = {
        "pc_id": int(fname) if fname.isdigit() else None,
        "labels": unique_labels,
        "name": fname,
    }
    with open(os.path.join(meta_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    print(f" → Saved {len(tiles)} tiles and metadata")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ply_files = glob.glob(os.path.join(DATA_DIR, "*.ply"))
    for ply_path in ply_files:
        process_ply(ply_path, OUT_DIR)


if __name__ == "__main__":
    main()