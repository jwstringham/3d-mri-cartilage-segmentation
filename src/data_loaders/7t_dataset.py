# 7t_dataset.py
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def load_7t_volumes(h5_path, slices_per_volume=80):
    """Load multiple 7T knee MRI volumes from HDF5, each with slices_per_volume slices."""
    with h5py.File(h5_path, "r") as f:
        slice_keys = sorted([k for k in f.keys() if k.startswith('Slice')])
        total_slices = len(slice_keys)
        
        volumes = []
        segmentations = []
        
        for start_idx in range(0, total_slices, slices_per_volume):
            end_idx = min(start_idx + slices_per_volume, total_slices)
            vol_slices = []
            seg_slices = []
            
            for i in range(start_idx, end_idx):
                key = slice_keys[i]
                img = np.array(f[key]['normalizedImage'])
                mask = np.array(f[key]['exportedSegMask'])
                vol_slices.append(img)
                seg_slices.append(mask)
            
            # Stack into 3D: (slices_per_volume, 512, 512)
            vol = np.stack(vol_slices, axis=0)
            seg = np.stack(seg_slices, axis=0)
            
            volumes.append(vol)
            segmentations.append(seg)
    
    return volumes, segmentations


class Knee7TDataset(Dataset):
    """
    7T knee MRI dataset. Loads multiple volumes from HDF5, each with slices_per_volume slices.
    Extracts random 3D patches from any volume with optional augmentation.
    """

    def __init__(self, h5_path, slices_per_volume=80, patch_size=(64, 128, 128), num_patches=1000, normalize=True, augment=False):
        self.h5_path = h5_path
        self.slices_per_volume = slices_per_volume
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.normalize = normalize
        self.augment = augment

        # Load all volumes
        self.volumes, self.segmentations = load_7t_volumes(h5_path, slices_per_volume)
        
        if self.normalize:
            # Normalize each volume separately
            for i in range(len(self.volumes)):
                vol = self.volumes[i]
                m = vol.mean()
                s = vol.std()
                if s > 0:
                    self.volumes[i] = (vol - m) / s

        self.num_volumes = len(self.volumes)
        # Each volume shape: (slices_per_volume, 512, 512)
        self.D = slices_per_volume
        self.H, self.W = 512, 512

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        # Randomly select a volume
        vol_idx = np.random.randint(0, self.num_volumes)
        vol = self.volumes[vol_idx]
        seg = self.segmentations[vol_idx]

        # Random patch
        d_patch, h_patch, w_patch = self.patch_size

        d_start = np.random.randint(0, self.D - d_patch + 1)
        h_start = np.random.randint(0, self.H - h_patch + 1)
        w_start = np.random.randint(0, self.W - w_patch + 1)

        vol_patch = vol[d_start:d_start+d_patch, h_start:h_start+h_patch, w_start:w_start+w_patch]
        seg_patch = seg[d_start:d_start+d_patch, h_start:h_start+h_patch, w_start:w_start+w_patch]

        if self.augment:
            # Random flip along spatial dimensions
            if np.random.rand() > 0.5:
                vol_patch = np.flip(vol_patch, axis=1)  # flip H
                seg_patch = np.flip(seg_patch, axis=1)
            if np.random.rand() > 0.5:
                vol_patch = np.flip(vol_patch, axis=2)  # flip W
                seg_patch = np.flip(seg_patch, axis=2)

        # Add channel dimension -> (1, d_patch, h_patch, w_patch)
        vol_patch = np.expand_dims(vol_patch, 0)

        return torch.from_numpy(vol_patch), torch.from_numpy(seg_patch)


if __name__ == "__main__":
    ds = Knee7TDataset(root="7T_Data/Neal_7T_Cartilages_20200504.hdf5")
    img, seg = ds[0]
    print("Image shape:", img.shape)
    print("Seg shape:", seg.shape, "unique labels:", torch.unique(seg))