#!/usr/bin/env python3
import os
import argparse
import h5py
import numpy as np
import torch

from ..models import vnet
from ..data_loaders import load_7t_volumes

def preprocess_volume(vol, normalize=True):
    # vol: (D, H, W)
    if normalize:
        m = float(vol.mean())
        s = float(vol.std())
        if s > 0:
            vol = (vol - m) / s

    # Add channel and batch dims: (1, 1, D, H, W)
    vol = np.expand_dims(vol, axis=(0, 0))
    return torch.from_numpy(vol)

def strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to vnet_model_best.pth.tar")
    ap.add_argument("--data", default='7T_Data/Neal_7T_Cartilages_20200504.hdf5', type=str,
                    help="path to 7T HDF5 data file (default: 7T_Data/Neal_7T_Cartilages_20200504.hdf5)")
    ap.add_argument("--out", required=True, help="output path prefix for predicted segmentations (e.g., pred_7t)")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--slices-per-volume", type=int, default=80, help="Number of slices per volume")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Load all volumes
    h5_path = args.data
    volumes, _ = load_7t_volumes(h5_path, args.slices_per_volume)
    
    # Build model
    num_classes = 9
    model = vnet.VNet(elu=False, nll=True, num_classes=num_classes).to(device)
    model.eval()

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)

    # Process each volume
    for i, vol in enumerate(volumes):
        D, H, W = vol.shape
        x = preprocess_volume(vol, normalize=(not args.no_normalize)).to(device)

        with torch.no_grad():
            output = model(x)  # (1, D*H*W, C)
            pred_flat = torch.argmax(output, dim=1)  # (D*H*W,)

        pred_3d = pred_flat.view(D, H, W).cpu().numpy().astype(np.uint8)

        # Save
        out_path = f"{args.out}_{i:02d}.h5"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("data", data=pred_3d, compression="gzip")
        print(f"Saved predicted segmentation for volume {i} to {out_path}, shape {pred_3d.shape}")

if __name__ == "__main__":
    main()