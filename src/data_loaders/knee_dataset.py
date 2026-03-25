# knee_dataset.py
import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def load_volume(path):
    """Load a knee MRI volume from an HDF5 .im file."""
    with h5py.File(path, "r") as f:
        vol = np.array(f["data"])  # e.g. (384, 384, 160) or similar
    return vol.astype(np.float32)


def load_segmentation(path):
    """Load cartilage segmentation labels from an HDF5 .seg file."""
    with h5py.File(path, "r") as f:
        seg = np.array(f["data"])
    return seg.astype(np.int16)


class KneeMRIDataset(Dataset):
    """
    Generic 3D knee MRI + segmentation dataset.

    Expects:
      root/
        train/ or valid/ or test/
          <split>_XXX_VYY.im
          <split>_XXX_VYY.seg  (for train/valid)
    """

    def __init__(self, root, split, normalize=True):
        assert split in ("train", "valid", "test")
        self.root = root
        self.split = split
        self.normalize = normalize

        split_dir = os.path.join(root, split)
        pattern = os.path.join(split_dir, f"{split}_*_V*.im")
        im_paths = sorted(glob.glob(pattern))

        self.samples = []
        for im_path in im_paths:
            base = os.path.splitext(os.path.basename(im_path))[0]  # e.g. valid_001_V00
            seg_path = os.path.join(split_dir, base + ".seg")
            if os.path.exists(seg_path):
                # train/valid
                self.samples.append((im_path, seg_path))
            else:
                # allow missing seg for test
                if split == "test":
                    self.samples.append((im_path, None))

        if len(self.samples) == 0:
            raise RuntimeError(f"No volumes found in {split_dir} with pattern {pattern}")

    def __len__(self):
        return len(self.samples)

    def _preprocess_image(self, vol):
        # vol: numpy array (D, H, W) or (H, W, D); whatever is in the HDF5
        vol = vol.astype(np.float32)
        if self.normalize:
            m = vol.mean()
            s = vol.std()
            if s > 0:
                vol = (vol - m) / s
        # Add channel dimension -> (1, D, H, W)
        vol = np.expand_dims(vol, 0)
        return vol

    def _preprocess_seg(self, seg):
        """
        seg: numpy array, currently (D, H, W, 6) with one-hot channels.
        We convert to a single-label volume:
          0 = background
          1..6 = cartilage compartments
        """
        seg = seg.astype(np.int64)

        # If shape has channels last (D, H, W, C):
        if seg.ndim == 4 and seg.shape[-1] == 6:
            # sum over channels to detect background
            summed = seg.sum(axis=-1)                  # (D, H, W), values in {0,1,...}
            # argmax over channels gives index 0..5
            argmax = seg.argmax(axis=-1) + 1          # shift to 1..6
            # where summed == 0 → background (0), else use argmax+1
            label_map = np.where(summed > 0, argmax, 0)
            seg = label_map
        # If it's already (D,H,W), we just ensure integer type
        elif seg.ndim == 3:
            # e.g., already integer labels 0..K
            pass
        else:
            raise RuntimeError(f"Unexpected seg shape: {seg.shape}")

        return seg  # (D, H, W), int64

    def __getitem__(self, idx):
        im_path, seg_path = self.samples[idx]

        vol = load_volume(im_path)
        vol = self._preprocess_image(vol)
        vol = torch.from_numpy(vol)  # (1, D, H, W)

        if seg_path is None:
            # For pure inference / test without labels
            return vol, None

        seg = load_segmentation(seg_path)
        seg = self._preprocess_seg(seg)
        seg = torch.from_numpy(seg)  # (D, H, W)

        return vol, seg


if __name__ == "__main__":
    ds = KneeMRIDataset(root="data", split="valid")
    img, seg = ds[0]
    print("Image shape:", img.shape)
    print("Seg shape:", seg.shape, "unique labels:", torch.unique(seg))
