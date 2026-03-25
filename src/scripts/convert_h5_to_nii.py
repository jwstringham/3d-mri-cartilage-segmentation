import re
from pathlib import Path

import h5py
import numpy as np
import nibabel as nib

SLICE_GROUP_RE = re.compile(r"^Slice\d+$")

def decode_h5_string(x) -> str:
	# x might be bytes, numpy bytes_, or an array of them
	if isinstance(x, (bytes, np.bytes_)):
		return x.decode("utf-8", errors="ignore")
	return str(x)

def sanitize_filename(name: str) -> str:
	# make it filesystem-friendly
	name = name.strip()
	name = name.replace("\\", "_").replace("/", "_")
	name = re.sub(r"\s+", "_", name)
	name = re.sub(r"[^A-Za-z0-9_.-]", "", name)
	return name or "unknown"

def save_nifti(vol, out_path, dtype, spacing=(1.0, 1.0, 4.0)):
	vol = vol.astype(dtype, copy=False)
	vol = np.transpose(vol, (1, 0, 2))  # (x, y, z)

	affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
	img = nib.Nifti1Image(vol, affine)
	img.header.set_zooms(spacing)

	nib.save(img, str(out_path))
	print("saved ->", out_path, "shape:", vol.shape, "zooms:", img.header.get_zooms())

def convert_h5_grouped_by_filename(h5_path: str, out_dir: str = "nii_out") -> None:
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	images = {}  # filename -> list of (slice_idx, img2d)
	masks = {}   # filename -> list of (slice_idx, mask2d)

	with h5py.File(h5_path, "r") as hf:
		key_list = list(hf.keys())

		# Notebook does: for key in keyList[3:-1]:
		# That implies first 3 are metadata (ContourId/Label/LabelNames),
		# last is StructureId.
		slice_keys = key_list[3:-1]

		for key in slice_keys:
			if not SLICE_GROUP_RE.match(key):
				continue

			grp = hf[key]

			# required datasets
			if "filename" not in grp or "slice" not in grp or "normalizedImage" not in grp:
				continue

			# filename stored as (1,) bytes
			fn_raw = np.array(grp["filename"]).reshape(-1)[0]
			fn = decode_h5_string(fn_raw).strip()

			# slice stored as (1,) int
			slice_idx = int(np.array(grp["slice"]).reshape(-1)[0])

			img2d = np.array(grp["normalizedImage"])
			if img2d.ndim != 2:
				continue

			images.setdefault(fn, []).append((slice_idx, img2d))

			# mask is optional but appears in your file
			if "exportedSegMask" in grp:
				mask2d = np.array(grp["exportedSegMask"])
				if mask2d.ndim == 2:
					masks.setdefault(fn, []).append((slice_idx, mask2d))

	print("Found volumes:", len(images))

	# build and save volumes
	for fn, slices_list in images.items():
		slices_list.sort(key=lambda t: t[0])
		vol = np.stack([t[1] for t in slices_list], axis=2)  # (rows, cols, z)

		base = sanitize_filename(fn)
		save_nifti(vol, out_dir / f"{base}_image.nii.gz", np.float32, spacing=(1.0, 1.0, 4.0))

		if fn in masks:
			masks[fn].sort(key=lambda t: t[0])
			seg = np.stack([t[1] for t in masks[fn]], axis=2)
			save_nifti(seg, out_dir / f"{base}_seg.nii.gz", np.uint8)

if __name__ == "__main__":
	convert_h5_grouped_by_filename(
		"7T_Data/Neal_7T_Cartilages_20200504.hdf5",
		out_dir="nii_7T_out"
	)