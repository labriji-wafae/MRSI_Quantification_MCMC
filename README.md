# MRSI Quantification Pipeline (MCMC & FSL-MRS)

An advanced, high-performance Python pipeline for **Magnetic Resonance Spectroscopy Imaging (MRSI)** voxel-wise quantification. This project integrates custom robust Bayesian inference algorithms, utilizing **Gibbs Sampling with Metropolis-Hastings within Gibbs**, alongside standard classical optimization routines from the University of Oxford's **FSL-MRS** framework (Newton & Metropolis-Hastings).

This software is designed to transform raw spatial complex MRS time-domain or frequency-domain signals into accurate, robust metabolite concentrations maps (e.g., NAA, Cho, Cr), featuring specialized automated tools for digital shimming alignment and polynomial baseline correction.

---

## 📖 Scientific Context & Publication

This work introduces a novel Bayesian model that explicitly accounts for potential errors in the observation linear operator during MRS restoration. The algorithm has been successfully validated on both synthetic databases and **clinical datasets from high-grade brain tumor glioblastoma (GBM) patients**.

### Citation
If you use this code or methods in your research, please cite our IEEE ISBI paper:
```text
@inproceedings{labriji2024bayesian,
  title={A Novel Bayesian Approach for Magnetic Resonance Spectroscopy Restoration with Operator Error Modeling},
  author={Labriji, Wafae and et al.},
  booktitle={2024 IEEE International Symposium on Biomedical Imaging (ISBI)},
  year={2024},
  organization={IEEE}
}

---
## 📐 Project Architecture

The codebase follows a standard production layout:

```text
MRSI_Quantification_MCMC/
│
├── src/                        # Core algorithmic package
│   ├── __init__.py
│   ├── mcmc.py                 # Bayesian Gibbs sampling & Bernoulli-Laplace sparse models
│   ├── tools_mcmc.py           # Numerical shimming alignment & direct polynomial fits
│   ├── pipelines.py            # Voxel loop orchestrators & distributed starmap pipelines
│   ├── utils.py                # Core I/O helper functions & NIfTI naming schemes
│   └── rtnorm.py               # Optimized Truncated Normal distribution sampler
│
├── notebooks/
│   └── Synthetic_data_gen_TE_139ms.ipynb  # Synthetic database exploration & validation
│
├── run_quantify.py             # Single clean CLI execution entrypoint (Argparse managed)
├── .gitignore                  # Production Git exclusion rules (.pyc, cache, checkpoints)
└── requirements.txt            # Python dependencies package

```

---
## ⚡ Mathematical & Signal Processing Highlights

* **Sparse Regularization via $L_0$-Norm Approximation:** The core innovation relies on a hierarchical Bayesian model incorporating a **Bernoulli-Laplace sparse prior** on metabolite concentrations. Activating a pseudo-$L_0$ regularization naturally drives non-significant metabolite amplitudes strictly to zero. This mathematical constraint allows users to safely expand the input metabolite basis set without the clinical risk of overfitting or quantifying pure background spectral noise.
* **Robust Bayesian Inference:** By utilizing a tailored Gibbs Sampler (with Metropolis-Hastings within Gibbs updates), the algorithm explores the joint posterior probability distribution rather than relying on point-estimates. This guarantees global numerical convergence and prevents the optimization path from falling into local minima traps common in complex MRS phase and damping landscapes.

---

## 🚀 Installation & Environment Setup

Due to advanced dependencies on neuroimaging medical ecosystems, setting up via a **Conda environment** is highly recommended.

### 1. Clone the repository

```bash
git clone [https://github.com/wafae-labriji/MRSI_Quantification_MCMC.git](https://github.com/your-username/MRSI_Quantification_MCMC.git)
cd MRSI_Quantification_MCMC

```

### 2. Create the environment & install FSL-MRS

Following standard Oxford University guidelines, create a pristine environment containing FSL-MRS and standard data toolkits:

```bash
conda create --name mrsi-env -c conda-forge -c fsl fsl_mrs python=3.10 -y
conda activate mrsi-env

```

### 3. Install Python requirements

Install remaining processing and structural dependencies directly:

```bash
pip install -r requirements.txt

```

---

## 💻 How to Run the Pipeline

The script `run_quantify.py` automatically scans a root directory, seeks any study folder named `CSI`, parses internal raw inputs ending with `_hsvd.nii` (NIfTI MRS formats, water suppressed), corrects frequency shifts, and launches active fitting models.

### Basic Usage (Runs all methods sequentially)

```bash
python run_quantify.py /path/to/your/study_root_directory

```

### Advanced Usage (CLI Flags Control)

You can selectively toggle fitting pipelines based on computation budgets. For instance, to execute only the custom Gibbs Sampler, disabling Newton or classical FSL methods:

```bash
python run_quantify.py /path/to/data --no-newton --no-mh --mcmc

```

### Discover all options:

```bash
python run_quantify.py --help

```

---

## 📊 Outputs & Result Formatting

All generated outputs are securely exported in standard clinical or processing extensions inside a dynamically checked `Results_test/` subfolder:

1. **`.nii.gz` volumes:** Spatial metabolic maps (e.g., `A_MCMC_*.nii.gz`) and computed Diagnostic Ratios like **CNI** (Choline-to-NAA Ratio maps) mapped directly back into the patient world coordinate space via world affine transformations.
2. **`.h5` HDF5 DataStores:** Structured Hierarchical Data Format (HDF5) containers managing different data structures per method:
   * **For FSL Methods:** Stores a collection of per-voxel tabular DataFrames containing classical optimization statistics, parameter uncertainties, and standard FSL Cramér-Rao Lower Bounds (CRLB).
   * **For Custom MCMC:** Stores high-dimensional matrices tracking full Markov Chain sampling iterations (e.g., lineshape damping chains, noise variance tracking, and posterior concentration distributions) for advanced statistical auditing.

```
