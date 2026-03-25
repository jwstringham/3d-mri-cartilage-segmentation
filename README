# V-Net PyTorch Implementation for Knee Cartilage Segmentation

This project implements V-Net, a fully convolutional neural network for volumetric medical image segmentation, adapted for knee cartilage segmentation using both low-resolution and high-resolution (7T) MRI data.

## Project Structure

```
VNET/
├── src/
│   ├── data_loaders/
│   │   ├── knee_dataset.py      # Low-res data loader
│   │   └── 7t_dataset.py        # High-res (7T) data loader
│   ├── inference/
│   │   ├── infer_knee.py        # Low-res inference
│   │   └── infer_7t.py          # High-res inference
│   ├── models/
│   │   └── vnet.py              # V-Net architecture
│   ├── training/
│   │   ├── train.py             # Low-res training
│   │   └── train_7t.py          # High-res training
│   ├── scripts/
│   │   └── convert_h5_to_nii.py # HDF5 to NIfTI conversion
│   └── utils/
│       ├── local.py
│       ├── make_graph.py
│       ├── plot.py
│       └── ...
├── 7T_Data/
│   └── Neal_7T_Cartilages_20200504.hdf5  # High-res data
├── data/                    # Low-res data directory (not included)
├── trained_models/          # Saved model checkpoints
├── results/                 # Segmentation results
├── outputs_for_viewing/     # Visualization outputs
├── requirements.txt         # Python dependencies
├── data_loader.ipynb        # Low-res data exploration
├── 7Tdataloader.ipynb       # High-res data exploration
└── README.md                # Project description
```

## Installation

### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd VNET
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The `requirements.txt` lists `matplotlib` (not `matplotlib.pyplot`). If you encounter import issues, ensure all packages are installed.

### 4. Verify PyTorch Installation

```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Data Path Configuration

All training and inference scripts support configurable data paths to accommodate different environments (local vs HPC):

### Low-Resolution Data
- **Training**: `--data /path/to/lowres/data/directory`
- **Inference**: `--im /path/to/input/file.im`

### High-Resolution Data  
- **Training**: `--data /path/to/7T_data.hdf5`
- **Inference**: `--data /path/to/7T_data.hdf5`

**Default Paths:**
- Low-res: `--data data` (expects `data/train/`, `data/valid/`, `data/test/`)
- 7T: `--data 7T_Data/Neal_7T_Cartilages_20200504.hdf5`

This allows you to:
- Use absolute paths for HPC environments
- Use relative paths for local development
- Mount data directories at different locations

## Training

### Low-Resolution Training

```bash
cd src/training
python train.py --data /path/to/data --save work/vnet.lowres --nEpochs 300 --batchSz 4
```

**Key Parameters:**
- `--data`: Path to data directory (default: 'data')
- `--save`: Output directory for logs and checkpoints
- `--nEpochs`: Number of training epochs (default: 300)
- `--batchSz`: Batch size (default: 10, reduce for memory)
- `--ngpu`: Number of GPUs (default: 1)
- `--opt`: Optimizer ('adam', 'sgd', 'rmsprop')

**Output:**
- `work/vnet.lowres/train.csv`: Training loss/error logs
- `work/vnet.lowres/test.csv`: Validation logs
- `work/vnet.lowres/vnet_model_best.pth.tar`: Best model checkpoint

### High-Resolution (7T) Training

```bash
cd src/training
python train_7t.py --data /path/to/7T_data.hdf5 --save work/vnet.7t --nEpochs 300 --batchSz 2
```

**Key Parameters:**
- `--data`: Path to 7T HDF5 file (default: '7T_Data/Neal_7T_Cartilages_20200504.hdf5')
- Same as low-res, but uses smaller batch size due to larger patches

**Features:**
- Trains on 10,000 random patches per epoch from 14 volumes
- Includes data augmentation (random flips)
- Each patch: 64×128×128 voxels

**Output:** Same format as low-res training.

## Inference

### Low-Resolution Inference

```bash
cd src/inference
python infer_knee.py --ckpt path/to/model_best.pth.tar --im data/test/test_001_V00.im --out pred_test_001.seg
```

**Parameters:**
- `--ckpt`: Path to trained model checkpoint
- `--im`: Input image file (.im)
- `--out`: Output segmentation file (.seg or .h5)
- `--device`: 'cuda' or 'cpu'

### High-Resolution (7T) Inference

```bash
cd src/inference
python infer_7t.py --data /path/to/7T_data.hdf5 --ckpt path/to/model_best.pth.tar --out pred_7t
```

**Parameters:**
- `--data`: Path to 7T HDF5 file (default: '7T_Data/Neal_7T_Cartilages_20200504.hdf5')
- `--ckpt`: Path to trained 7T model checkpoint
- `--out`: Output prefix (will create pred_7t_00.h5 through pred_7t_13.h5)
- `--device`: 'cuda' or 'cpu'
- `--slices-per-volume`: Slices per volume (default: 80)

**Output:** 14 segmentation files, one for each 80-slice volume.

## Evaluation

### Training Curves

After training, plot the loss and error curves:

```bash
cd src/utils
python plot.py 10 work/vnet.lowres  # 10 batches per epoch average
```

This generates `work/vnet.lowres/loss.png` and `work/vnet.lowres/error.png`.

### Quantitative Metrics

The training scripts output:
- **Loss**: NLL loss
- **Error**: Pixel-wise accuracy (%)
- **Dice**: Mean Dice coefficient across classes

For 7T: 9 classes (0=background, 1-8=cartilage/bone)
For low-res: 7 classes (0=background, 1-6=cartilage)

## Model Architecture

V-Net is a 3D U-Net variant with:
- Encoder-decoder structure
- 4 resolution levels
- Batch normalization and optional dropout
- ELU/PReLU activations
- Output: Log-probabilities for NLL loss

**Input:** (N, 1, D, H, W)
**Output:** (N*D*H*W, num_classes)

## Data Exploration

### Low-Resolution Data

Use `data_loader.ipynb` to explore low-res data:
- Visualize individual slices
- Examine segmentation masks
- Check data dimensions and labels

### High-Resolution Data

Use `7Tdataloader.ipynb` to explore 7T data:
- Inspect HDF5 structure
- Visualize high-res slices
- Check label distributions

## Utilities

### HDF5 to NIfTI Conversion

Convert HDF5 data to NIfTI format for external viewers:

```bash
cd src/scripts
python convert_h5_to_nii.py
```

### Model Graph Visualization

Generate computational graph (requires graphviz):

```bash
cd src/utils
python make_graph.py
```

## Troubleshooting

### Memory Issues
- Reduce batch size (`--batchSz`)
- Use smaller patches for 7T training
- Switch to CPU if GPU memory is insufficient

### Import Errors
- Ensure virtual environment is activated
- Check PyTorch CUDA compatibility
- Reinstall packages: `pip install -r requirements.txt --force-reinstall`

### Data Loading Issues
- Verify data paths exist
- Check HDF5 file integrity
- Ensure proper file permissions

## Performance Tips

### Training
- Use multiple GPUs for faster training (`--ngpu 2`)
- Monitor GPU utilization with `nvidia-smi`
- Adjust learning rate schedule in `adjust_opt()`

### Inference
- Use CUDA for faster inference
- Process volumes sequentially to manage memory
- Consider sliding window for very large volumes

## Citation

If you use this implementation, please cite the original V-Net paper:

```
@article{milletari2016v,
  title={V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation},
  author={Milletari, Fausto and Navab, Nassir and Ahmadi, Seyed-Ahmad},
  journal={arXiv preprint arXiv:1606.04797},
  year={2016}
}
```

## License

See LICENSE file for details.
