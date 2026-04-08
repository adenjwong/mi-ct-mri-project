# Mutual Information CT–MRI Spine Registration Project

## Overview

This project investigates **rigid multimodal medical image registration** between **CT** and **MRI** volumes of the spine using **Mutual Information (MI)** as the similarity metric. The main goal is not simply to produce a working registration pipeline, but to study a specific mathematical question:

**How do probability density estimation choices affect the optimization behavior of MI-based CT–MRI registration?**

More specifically, this project compares how different **histogram bin counts** and different MI implementations influence:

- convergence behavior,
- robustness to initialization,
- final alignment quality, and
- the smoothness/stability of the optimization landscape.

The project is motivated by the classical paper:

> W. M. Wells III, P. Viola, H. Atsumi, S. Nakajima, and R. Kikinis, “Multi-modal volume registration by maximization of mutual information,” *IEEE Transactions on Medical Imaging*, vol. 15, no. 1, pp. 35–45, 1996.

In addition to rigid registration, this project now includes an optional **deformable (non-rigid) registration extension using B-spline free-form deformation (FFD)**. This extension is designed as a secondary experimental component to investigate how increasing transformation flexibility affects Mutual Information-based optimization and alignment quality.

This repository is designed as a **small, mathematically focused experimental framework** suitable for a course project, mini paper, or technical presentation.

---

## Problem Space

### Why CT–MRI registration matters

CT and MRI provide complementary anatomical information:

- **CT** provides excellent visualization of **bone anatomy** and high spatial resolution for osseous structures.
- **MRI** provides strong **soft tissue contrast**, making it better for neural structures, discs, spinal cord, and surrounding soft tissues.

In spine imaging, both modalities are valuable. For example:

- CT can show vertebral anatomy and bony narrowing clearly.
- MRI can show nerve roots, discs, cord compression, and soft-tissue pathology.

However, these images are acquired from different scanners, at different times, with different physics, different intensity scales, and often different patient positioning. If they are to be compared directly or fused into one common coordinate system, they must first be **registered**.

### The registration challenge

For single-modality registration, simple similarity measures such as cross-correlation or sum of squared differences may work because corresponding anatomy often has similar intensity patterns. This is not true for CT and MRI.

A vertebra may appear:

- very bright in CT,
- relatively dark in MRI.

CSF or soft tissues may invert their appearance relative to CT. As a result, multimodal registration requires a metric that does **not** assume direct intensity similarity.

### Why Mutual Information

Mutual Information is attractive because it measures **statistical dependence** between the intensity distributions of two images rather than direct intensity equality. If CT and MRI are correctly aligned, corresponding anatomical structures generate a more structured joint intensity distribution, and MI tends to increase.

This makes MI one of the foundational similarity measures for multimodal registration.

---

## Mathematical Scope of the Project

This repository focuses on the following mathematical chain:

1. **Registration as an optimization problem**
2. **Mutual Information as the objective function**
3. **Probability density estimation for entropy and joint entropy**
4. **How density estimation changes the optimization landscape**

The project does **not** try to solve all problems in multimodal registration. Instead, it isolates a manageable and interesting question:

### Core question

How do the following choices influence MI-based rigid CT–MRI registration?

- number of histogram bins,
- metric variant,
- smoothing behavior implicit in the MI implementation,
- initialization of the rigid transform.

### Intended deliverables

This project is designed to support:

- a working experimental registration pipeline,
- metric convergence curves,
- before/after overlay visualizations,
- a summary table of experiment runs,
- discussion of optimization stability and sensitivity,
- a mathematically grounded short paper or presentation.

---

## Scope of the Current Implementation

### Included in this repository

- 3D **rigid** CT–MRI registration
- SimpleITK-based implementation
- comparison of MI metric variants available in SimpleITK
- comparison across multiple histogram bin counts
- repeated runs with different random perturbations of initialization
- saving registered volumes, transforms, plots, and tabular summaries
- post-hoc computation of MI and NMI for analysis

### Not included by default

- full deformable registration experiments
- custom hand-written MI optimizer
- deep learning registration
- automatic DICOM folder parsing
- batch multi-patient metadata management
- advanced preprocessing such as bias field correction, cropping, or segmentation-based masking

### Stretch goal supported conceptually

A future extension of this project is **non-rigid registration using B-spline free-form deformation**, with MI or NMI-style objectives plus regularization. That extension is not the main implementation target of this repository, but the current code structure is organized so that it can be added later.

### Deformable registration (extension)

This repository also includes an optional **deformable registration stage** implemented using B-spline transforms in SimpleITK.

This extension:

- runs after rigid registration as a refinement step
- is initialized using the rigid transform
- allows investigation of local anatomical alignment
- is intended for controlled experiments, not production use

This enables comparison between:

- rigid-only registration
- rigid + deformable registration

---

## Project Goals

The practical goals of this codebase are:

1. Load one CT volume and one MRI volume.
2. Preprocess them in a simple, reproducible way.
3. Register MRI to CT using a rigid transform and MI-based similarity.
4. Save the final registered MRI and transform.
5. Record per-iteration metric values.
6. Compare multiple settings across repeated runs.
7. Produce outputs suitable for figures, tables, and discussion.

The scientific goals are:

1. Study how MI behaves under different probability estimation settings.
2. Compare metric implementations such as Mattes MI and joint-histogram MI.
3. Study robustness to initialization.
4. Evaluate whether smoother metric behavior leads to more stable optimization.

---

## Repository Structure

A suggested repository structure is shown below.

```text
mi_ct_mri_project/
│
├── data/
│   ├── ct/
│   ├── mri/
│   └── metadata/
│
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── register_rigid.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── experiments.py
│   └── run_one_case.py
│   ├── register_deformable_bspline.py
│   ├── run_deformable_case.py
│   ├── inspect_images.py
│
├── results/
│   ├── transforms/
│   ├── metrics/
│   ├── plots/
│   └── overlays/
│
├── refs/
├── presentation/
├── requirements.txt
├── environment.yml
└── README.md
```

### Directory purpose

#### `data/`
Holds all input data.

- `data/ct/`: fixed CT volumes
- `data/mri/`: moving MRI volumes
- `data/metadata/`: optional notes, patient mapping, or experiment manifests

#### `src/`
Contains all source code.

#### `results/`
Stores generated outputs such as registered images, plots, metrics, transforms, and summaries.

#### `refs/`
Optional folder for PDFs, citation notes, or paper summaries.

#### `presentation/`
Optional folder for presentation slides, figures, and talking points.

---

## Expected Input Data

### Required input

The current implementation expects:

- **one 3D CT volume file**
- **one 3D MRI volume file**

The simplest supported formats are:

- `.nii`
- `.nii.gz`
- `.mha`
- `.mhd`

### Recommended assumptions

The code works best if:

- CT and MRI are from the **same patient**
- both cover roughly the **same anatomical region**
- both are approximately in the correct physical orientation
- spacing, origin, and direction metadata are valid

### Fixed vs moving image convention

This repository assumes:

- **CT = fixed image**
- **MRI = moving image**

The registration estimates a transform that brings the MRI into CT space.

### DICOM note

The current code is written for direct image files such as NIfTI. It does **not** yet include a dedicated DICOM-series loader for full folder-based DICOM input. If your data is stored as DICOM slices in folders, you will need either:

- conversion to NIfTI beforehand, or
- a custom DICOM series loading function.

---

## Environment Setup

Two setup methods are provided:

1. **Conda environment** via `environment.yml`
2. **Python virtual environment** via `requirements.txt`

### Option 1: Conda

Create the environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate mi-ct-mri
```

If you update the environment later:

```bash
conda env update -f environment.yml --prune
```

### Option 2: Python virtual environment with pip

Create a local virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Minimal dependency list

The codebase currently depends on:

- `SimpleITK`
- `numpy`
- `matplotlib`
- `pandas`

Notebook support can optionally be added with:

- `jupyter`
- `ipykernel`

---

## Environment Files

### `requirements.txt`
Used for pip-based installation.

Typical contents:

```txt
SimpleITK>=2.3.1
numpy>=1.24
matplotlib>=3.7
pandas>=2.0
jupyter>=1.0
ipykernel>=6.0
```

### `environment.yml`
Used for conda-based installation.

Typical contents:

```yaml
name: mi-ct-mri
channels:
  - conda-forge
dependencies:
  - python=3.11
  - simpleitk>=2.3.1
  - numpy>=1.24
  - matplotlib>=3.7
  - pandas>=2.0
  - jupyter
  - ipykernel
  - pip
```

---

## Source Code Walkthrough

This section explains what each file does and how the code flows across the project.

### 1. `src/load_data.py`

#### Purpose
Responsible for reading image data and printing basic metadata.

#### Main functions

- `load_image(path)`
  - reads a medical image from disk using SimpleITK
  - casts it to `sitkFloat32`

- `load_fixed_moving(ct_path, mri_path)`
  - loads the CT and MRI images using the convention:
    - CT = fixed
    - MRI = moving

- `print_image_info(name, image)`
  - prints size, spacing, origin, direction, and pixel type
  - useful for debugging metadata problems

#### Why it matters
Registration is performed in physical space, so invalid metadata can cause strange behavior. This file makes it easy to inspect inputs early.

## Deformable Registration Extension

### Motivation

Rigid registration captures global alignment but cannot model:

- local anatomical variation
- curvature differences
- soft tissue deformation

To explore whether local flexibility improves multimodal registration, this project includes a **deformable refinement stage** using B-spline free-form deformation.

---

### Pipeline Structure

The full pipeline becomes:

CT (fixed) + MRI (moving)  
↓  
Rigid registration  
↓  
Deformable refinement (B-spline)

---

### Key Design Principle

Deformable registration is always initialized from the rigid result.

This is critical because:

- it improves convergence stability
- it prevents unrealistic warping
- it preserves meaningful global alignment

---

### Typical Parameters

Example deformable settings:

- mesh size: (4, 4, 3)
- iterations: 50–100
- metric: Mattes Mutual Information

---

### Experimental Questions

This extension enables investigation of:

- whether deformable registration improves MI/NMI
- whether rigid alignment is already sufficient
- when deformable registration overfits
- how increased flexibility affects optimization stability

---

### Important Note

Deformable registration introduces many more degrees of freedom.

This can:

- improve alignment in some cases
- degrade results in others

It is therefore treated as an **experimental extension**, not a guaranteed improvement.

---

### 2. `src/preprocess.py`

#### Purpose
Contains simple preprocessing functions to standardize images before registration.

#### Main functions

- `clamp_intensity(image, lower, upper)`
  - clamps CT intensity values to reduce the effect of extreme outliers

- `normalize_to_0_1(image)`
  - rescales intensities to the range `[0, 1]`

- `zscore_normalize(image)`
  - standardizes intensities using mean and standard deviation

- `resample_to_reference(moving, reference, ...)`
  - resamples the moving image onto the reference grid using the identity transform
  - useful when you want both images on comparable voxel grids before registration

- `preprocess_ct_mri(...)`
  - the main preprocessing entry point
  - applies clamping, normalization, and optional grid matching

#### Why it matters
Although MI does not require matched intensity ranges, some normalization still improves numerical stability and reproducibility.

---

### 3. `src/register_rigid.py`

#### Purpose
Implements the actual rigid registration pipeline.

#### Main responsibilities

- create an initial rigid transform
- optionally perturb initialization for robustness experiments
- configure the MI metric
- configure the optimizer
- run registration
- collect metric values at each iteration
- resample the registered MRI
- save transforms and result metadata

#### Main functions

- `_make_initial_transform(...)`
  - creates a centered 3D rigid transform (`Euler3D` or `VersorRigid3D`)

- `_perturb_transform(...)`
  - adds random rotational and translational perturbations
  - used for initialization robustness studies

- `_configure_metric(...)`
  - selects the registration metric
  - currently supports:
    - `mattes_mi`
    - `joint_hist_mi`

- `_configure_optimizer(...)`
  - configures a regular-step gradient descent optimizer

- `run_rigid_registration(...)`
  - the main registration function
  - executes the optimization and returns:
    - final transform
    - dictionary of run results and per-iteration metric values

- `resample_registered_image(...)`
  - resamples MRI into CT space after registration

- `save_transform(...)`
  - saves the final transform to disk

- `save_results_json(...)`
  - saves a JSON summary of registration metadata and convergence history

#### Why it matters
This is the core experimental engine of the repository.

---

### 4. `src/evaluate.py`

#### Purpose
Contains post-hoc evaluation and summary functions.

#### Main functions

- `_safe_histogram(...)`
  - computes a normalized histogram

- `_safe_joint_histogram(...)`
  - computes a normalized joint histogram

- `entropy_from_probabilities(p)`
  - computes entropy from a probability vector

- `compute_mi_from_images(image1, image2, bins)`
  - computes post-hoc Mutual Information from two images

- `compute_nmi_from_images(image1, image2, bins)`
  - computes post-hoc Normalized Mutual Information

- `dice_score(mask1, mask2)`
  - computes the Dice coefficient for binary masks

- `centroid_mm_from_mask(mask)`
  - computes the physical-space centroid of a binary mask

- `centroid_distance_mm(mask1, mask2)`
  - computes centroid distance between two masks

- `summarize_registration(...)`
  - creates a compact post-hoc summary for registered outputs

#### Why it matters
This file allows you to separate:

- the metric used during optimization,
- from the metrics used afterward for analysis.

It is especially helpful if you later want to add segmentation-based evaluation.

---

### 5. `src/visualize.py`

#### Purpose
Generates plots and visual outputs for analysis and presentation.

#### Main functions

- `save_metric_curve(metric_values, out_path, title)`
  - plots metric value vs iteration
  - useful for studying convergence

- `_extract_middle_slice(image, axis)`
  - extracts a representative middle slice from a 3D image

- `save_overlay_figure(fixed, moving_before, moving_after, out_path, axis)`
  - produces a three-panel visualization:
    - fixed CT
    - before registration overlay
    - after registration overlay

#### Why it matters
These outputs are useful for:

- debugging
- figure generation
- inclusion in reports and presentations

---

### 6. `src/experiments.py`

#### Purpose
Runs a grid of experiments across multiple metrics, bin counts, and random seeds.

#### Main responsibilities

- load and preprocess one CT/MRI pair
- loop over metric settings
- loop over histogram bin counts
- loop over random seeds
- run registration for each configuration
- save outputs for each run
- aggregate results into a summary CSV

#### Main outputs

For each run:

- registered MRI volume
- final transform
- registration JSON
- convergence curve
- overlay image
- compact run summary JSON

For the full experiment:

- `summary.csv`

#### Why it matters
This is the primary script for generating the data you will analyze in the paper.

---

### 7. `src/run_one_case.py`

#### Purpose
A simplified single-run entry point for debugging and initial testing.

#### Why it matters
Before running a full experiment grid, you should first confirm that:

- one case loads correctly,
- registration runs without error,
- outputs look reasonable.

This script is the fastest way to do that.

---

## Run Order

The recommended order of execution is intentionally conservative.

### Step 1: Set up the environment

Use either conda or venv as described above.

### Step 2: Place the input data

Example:

```text
data/
  ct/
    sample_ct.nii.gz
  mri/
    sample_mri.nii.gz
```

### Step 3: Run one debug case

This should always be your first execution step.

```bash
python src/run_one_case.py \
  --ct data/ct/sample_ct.nii.gz \
  --mri data/mri/sample_mri.nii.gz \
  --outdir results/test_run \
  --metric mattes_mi \
  --bins 32 \
  --iterations 200
```

### Step 4: Inspect the outputs

Check the following files in `results/test_run/`:

- `registered_mri.nii.gz`
- `final_transform.tfm`
- `registration_results.json`
- `metric_curve.png`
- `overlay.png`

At this stage, you want to confirm:

- the code ran successfully,
- the convergence curve looks reasonable,
- the overlay after registration is better than before registration.

### Step 5: Run the full experiment grid

Once the single-case run works, scale to a grid search.

```bash
python src/experiments.py \
  --ct data/ct/sample_ct.nii.gz \
  --mri data/mri/sample_mri.nii.gz \
  --outdir results/experiment_01 \
  --metrics mattes_mi joint_hist_mi \
  --bins 16 32 64 \
  --seeds 0 1 2 3 4 \
  --iterations 200 \
  --perturb_init
```

### Step 6: Analyze the summary table

The main summary file will be:

```text
results/experiment_01/summary.csv
```

This can be loaded into Python, Excel, or a notebook for further analysis.

---

## Command-Line Usage

### Single-case registration

```bash
python src/run_one_case.py \
  --ct data/ct/sample_ct.nii.gz \
  --mri data/mri/sample_mri.nii.gz \
  --outdir results/test_run \
  --metric mattes_mi \
  --bins 32 \
  --iterations 200 \
  --sampling_percentage 0.2 \
  --normalization 0_1
```

### Single-case registration with perturbed initialization

```bash
python src/run_one_case.py \
  --ct data/ct/sample_ct.nii.gz \
  --mri data/mri/sample_mri.nii.gz \
  --outdir results/test_run_seed0 \
  --metric mattes_mi \
  --bins 32 \
  --iterations 200 \
  --perturb_init \
  --seed 0
```

### Full experiment grid

```bash
python src/experiments.py \
  --ct data/ct/sample_ct.nii.gz \
  --mri data/mri/sample_mri.nii.gz \
  --outdir results/experiment_01 \
  --metrics mattes_mi joint_hist_mi \
  --bins 16 32 64 \
  --seeds 0 1 2 3 4 \
  --iterations 200 \
  --sampling_percentage 0.2 \
  --normalization 0_1 \
  --perturb_init
```

### Step 7: Deformable registration (optional)

After confirming rigid registration works, you can run deformable refinement:

```bash
python src/run_deformable_case.py \
  --ct data/ct/sample_ct.nii.gz \
  --mri data/mri/sample_mri.nii.gz \
  --outdir results/deformable_test \
  --metric mattes_mi \
  --bins 32 \
  --rigid_iterations 200 \
  --deformable_iterations 75 \
  --mesh_x 4 --mesh_y 4 --mesh_z 3 \
  --perturb_init \
  --seed 0
```

---

## Output Files and Their Meaning

Each run directory contains several outputs.

### `registered_mri.nii.gz`
The moving MRI image resampled into CT space after applying the final transform.

### `final_transform.tfm`
The final rigid transform estimated by the optimizer.

### `registration_results.json`
Contains run metadata such as:

- metric name
- bin count
- optimizer settings
- stop condition
- final parameter values
- per-iteration metric values

### `metric_curve.png`
A plot of metric value versus iteration. Useful for convergence analysis.

### `overlay.png`
A visual comparison of:

- fixed CT,
- overlay before registration,
- overlay after registration.

### `summary.json`
A compact summary of the run.

### `summary.csv`
Created only by the experiment runner. This is the aggregate table across all configurations.

### Additional outputs (deformable extension)

When deformable registration is used, additional outputs may include:

- deformably registered MRI volume
- deformable transform file
- deformable metric curve
- comparison summaries between rigid and deformable stages

---

## Workflow for the Course Project

A practical workflow for the course project is:

### Phase 1: Sanity check

- run a single case
- verify visual alignment improved
- confirm convergence curve is sensible

### Phase 2: Bin count study

Run:

- 16 bins
- 32 bins
- 64 bins

Compare:

- final metric values
- convergence smoothness
- visual alignment

### Phase 3: Initialization robustness study

For each bin count and metric:

- run multiple random seeds
- perturb the initial transform
- compare variability in final result

### Phase 4: Analysis

Generate tables and plots answering questions like:

- Does a smaller number of bins lead to smoother convergence?
- Are some settings more robust to poor initialization?
- Does the joint-histogram MI behave differently from Mattes MI?
- Do post-hoc MI/NMI trends agree with optimizer behavior?

---

## Interpretation of the Experiments

### Why compare bin counts?

Mutual Information depends on probability density estimation. Histogram binning discretizes intensities. The number of bins changes:

- estimator smoothness,
- estimator noise,
- entropy values,
- optimization surface shape.

This is the central mathematical focus of the project.

### Why compare metric variants?

SimpleITK exposes different MI-style metrics with different implementation details. Comparing them helps determine whether smoother probability estimation changes optimization stability.

### Why perturb initialization?

Registration is a non-convex optimization problem. Different initial transforms may lead to:

- different local optima,
- failed convergence,
- sensitivity to surface roughness.

This is a direct way to test robustness.

### Deformable registration interpretation

Deformable registration should not be assumed to always improve results.

Possible outcomes include:

- improvement in MI/NMI and alignment
- minimal change if rigid alignment is sufficient
- degradation due to overfitting or instability

This reflects a fundamental trade-off:

- increased flexibility vs decreased stability

---

## Common Failure Modes

### 1. Images do not overlap enough

If CT and MRI are too far apart initially, rigid registration may fail.

**Possible fixes:**

- crop to the same anatomical region,
- use better initial alignment,
- reduce perturbation size during testing.

### 2. Metadata mismatch

Spacing, origin, or direction may be incorrect.

**Possible fixes:**

- inspect with `print_image_info`,
- verify image headers,
- resave data consistently.

### 3. Noisy or unstable convergence

This may happen because:

- too many bins,
- low overlap,
- poor initialization,
- insufficient sampling.

**Possible fixes:**

- reduce bin count,
- increase sampling percentage,
- normalize images,
- improve initialization.

### 4. Overlay looks unchanged

This may indicate:

- registration is not improving,
- anatomy is too mismatched,
- visualization plane is not informative.

**Possible fixes:**

- inspect another slice axis,
- verify physical overlap,
- use additional evaluation metrics.

---

## Limitations

This repository is intentionally limited in several ways.

- It focuses on **one CT/MRI pair at a time**.
- It assumes **rigid registration** only.
- It does not include a formal landmark annotation system.
- It does not automate DICOM loading.
- It does not implement custom MI derivatives from scratch.
- It does not guarantee clinical-quality registration.

These limitations are intentional because the project is aimed at a mathematically focused experimental study, not a production clinical pipeline.

---

## Recommended First Run

If this is your first time using the project, use this exact sequence.

### Create the environment

```bash
conda env create -f environment.yml
conda activate mi-ct-mri
```

### Run a single case

```bash
python src/run_one_case.py \
  --ct data/ct/sample_ct.nii.gz \
  --mri data/mri/sample_mri.nii.gz \
  --outdir results/test_run \
  --metric mattes_mi \
  --bins 32 \
  --iterations 200
```

### Then inspect

- `results/test_run/overlay.png`
- `results/test_run/metric_curve.png`
- `results/test_run/registration_results.json`

Only after that should you run the experiment grid.

---

## Citation Context for the Project

Core references relevant to this repository include:

1. Wells et al., 1996 — original multimodal MI registration paper
2. Studholme et al., 1999 — normalized mutual information
3. Mattes et al., 2001 — practical MI estimation for registration
4. Pluim et al., 2003 — MI registration survey
5. Rueckert et al., 1999 — B-spline free-form deformation for non-rigid registration

These references motivate the current implementation and possible future extensions.

---

## Final Notes

This repository is meant to be:

- understandable,
- experimentally useful,
- mathematically motivated,
- easy to extend for a course project.

The most important idea behind the project is not simply that MI can register CT and MRI, but that the **way MI is estimated** changes the behavior of the optimization itself. This makes the project a useful bridge between:

- medical imaging,
- information theory,
- numerical optimization, and
- experimental computational analysis.
