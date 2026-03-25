#!/usr/bin/env python3
import os
import argparse
import h5py
import numpy as np
import torch

from ..models import vnet

def load_volume_h5(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        vol = np.array(f["data"])
    return vol.astype(np.float32)

def preprocess_like_training(vol: np.ndarray, normalize: bool = True) -> torch.Tensor:
    # training did: mean/std normalize + add channel dim
    if normalize:
        m = float(vol.mean())
        s = float(vol.std())
        if s > 0:
            vol = (vol - m) / s

    # Expect vol shape is (D,H,W) like your training logs imply (384,384,160)
    vol = np.expand_dims(vol, axis=0)   # (1,D,H,W)
    vol = np.expand_dims(vol, axis=0)   # (N=1,C=1,D,H,W)
    return torch.from_numpy(vol)

def strip_module_prefix(state_dict):
    # If saved from DataParallel, keys look like "module.xxx"
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to vnet_model_best.pth.tar")
    ap.add_argument("--im", required=True, help="path to input .im (HDF5) file")
    ap.add_argument("--out", required=True, help="output path for predicted labels (e.g., pred.npy)")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # 1) Build model exactly like training
    num_classes = 7
    model = vnet.VNet(elu=False, nll=True, num_classes=num_classes).to(device)
    model.eval()
    print("Evaluation Started")

    # 2) Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    print("Model Loaded")

    # Your save_checkpoint stored: {'epoch', 'state_dict', 'best_prec1'}
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)

    # 3) Load + preprocess input volume
    vol = load_volume_h5(args.im)
    x = preprocess_like_training(vol, normalize=(not args.no_normalize)).to(device)
    print("Preprocessing")

    # 4) Forward pass
    with torch.no_grad():
        # Your model returns log-probs flattened: (Nvoxels, C)
        out_flat = model(x)  # shape: (D*H*W, C)

        pred_flat = torch.argmax(out_flat, dim=1)  # (D*H*W,)

    print("Forward Pass Completed")

    # 5) Reshape back to 3D
    # x shape: (1,1,D,H,W)
    _, _, D, H, W = x.shape
    pred_3d = pred_flat.view(D, H, W).cpu().numpy().astype(np.uint8)

    # 6) Save output
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if out_path.endswith(".npy"):
        np.save(out_path, pred_3d)
        print(f"Saved predicted labels to {out_path} with shape {pred_3d.shape}")
    elif out_path.endswith(".h5") or out_path.endswith(".seg"):
        # HDF5 style (like your training segs): dataset key "data"
        with h5py.File(out_path, "w") as f:
            f.create_dataset("data", data=pred_3d, compression="gzip")
        print(f"Saved predicted labels to {out_path} (HDF5 dataset 'data'), shape {pred_3d.shape}")
    else:
        # default: save npy if extension unknown
        np.save(out_path + ".npy", pred_3d)
        print(f"Unknown extension; saved to {out_path}.npy, shape {pred_3d.shape}")

if __name__ == "__main__":
    main()
