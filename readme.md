# DL-WA-CSI: Deep Learning for Weighted Averaging in Chemical Shift Imaging

**DL-WA-CSI** is a Deep Learning framework designed to denoise and reconstruct high-quality **Chemical Shift Imaging (CSI)** data from low-signal-to-noise ratio (SNR) acquisitions. The core of this project uses a specialized **FCA-UNet** (Frequency-Channel Attention UNet) to map noisy, weighted-averaged spectra back to their high-fidelity ground truth.

## 🚀 Overview

The pipeline leverages on-the-fly simulation to generate training data. It combines anatomical priors (from the IXI dataset) with synthetic metabolic curves and chemical shifts to simulate realistic MRS/CSI datasets.

* **Model:** FCA-UNet (32x32 sample size, 72 spectral channels).
* **Simulation:** Powered by [MRSpy](https://gitee.com/txz32102/MRSpy), a custom-built library for magnetic resonance spectroscopy simulation and plotting.
* **Data Strategy:** Uses dynamic noise levels and random smooth metabolic curves to ensure model robustness.

---

## 📁 Project Structure

```bash
DL-WA-CSI/
├── data/               # Text files containing dataset paths
├── log/                # Training logs, loss curves, and epoch-wise visualizations
├── model/              # Architecture definitions (fca_unet.py)
├── util/               
│   ├── dataset.py      # Custom PyTorch Dataset (MRSDataset)
│   └── util.py         # Simulation logic and helper functions
├── train.py            # Main training and evaluation script
├── requirements.txt    # Project dependencies
└── readme.md           # You are here

```


## 📊 Data Format

The project expects `.txt` files (e.g., `hybrid_train.txt`) to manage the dataset. Each line in the text file should be an **absolute path** to a folder containing the required metabolite images.

### Folder Requirements

Each folder path listed in the `.txt` file must contain:

* `water.jpg`: Proton/Water reference spatial map.
* `glu.jpg`: Glutamate concentration map.
* `lac.jpg`: Lactate concentration map.

**Example `hybrid_train.txt`:**

```text
/home/user/data/subject_001/slice_162
/home/user/data/subject_002/slice_145

```


## 🛠️ Installation & Dependencies

This project requires **MRSpy**. Ensure you have it installed along with other dependencies:

```bash
# Clone and install MRSpy first
git clone https://gitee.com/txz32102/MRSpy
cd MRSpy
pip install -e .

# Install DL-WA-CSI requirements
cd ../DL-WA-CSI
pip install -r requirements.txt

```


## 🏋️ Training

The training script `train.py` performs on-the-fly simulation using the `util.simulation` function. It applies random noise levels ( to `max_noise`) and generates random chemical shifts to prevent overfitting.

To start training:

```bash
python train.py --epochs 40 --batch_size 3 --lr 0.0001 --device cuda:0 --noise_level 0.8
```

### Key Arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--train_txt_path` | `data/hybrid_train.txt` | Path to training folder list |
| `--noise_level` | `0.8` | Maximum random noise level for simulation |
| `--project_name` | `WA` | Name used for log subdirectories |
| `--device` | `cuda:0` | Target GPU device |


## 📈 Monitoring & Visualization

During training, the script automatically saves visualizations to the `log/` directory every epoch. It uses `SpecPlotter` from **MRSpy** to generate:

1. **Chemical Shift Images:** Spatial maps of metabolites.
2. **Spectral Plots:** Voxel-wise comparisons between Noisy (Weighted), Predicted, and Ground Truth spectra.
3. **Loss Logs:** `loss_log.txt` tracking MSE for train and test sets.

## 🔗 Related Library

This project relies heavily on **[MRSpy](https://gitee.com/txz32102/MRSpy)** for:

* `mrspy.sim.sim`: Simulating the CSI signal physics.
* `mrspy.plot`: High-quality spectral and spatial visualization.
* `mrspy.util`: Efficient image loading and tensor manipulation.
