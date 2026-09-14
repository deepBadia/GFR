

# SurMod

SurMod is a production-oriented Python package for **surrogate modeling**.

It provides a complete pipeline to train neural network surrogates that map design parameters and a continuous sweep variable (typically frequency) to complex or real-valued responses.

---

## Prerequisites

Before installing SurMod you must have:

- A working **Python** installation (3.9 or newer)
- A **Miniconda** (or Anaconda / Mamba) environment ready

Once those are in place, proceed with the installation steps below.

---

## Installation

### 1. Install PyTorch

```bash
pip install torch
```

(If you have specific hardware requirements such as a particular CUDA version, install the matching PyTorch build from the official PyTorch website first.)

### 2. Install SurMod

From a local wheel or source distribution:

```bash
# From a local .whl file
pip install path/to/surmod-0.x.x-py3-none-any.whl

# From a local .tar.gz file
pip install path/to/surmod-0.x.x.tar.gz
```

You can also install the default PyTorch at the same time by using the optional extra:

```bash
pip install path/to/surmod-0.x.x-py3-none-any.whl[torch]
```

### 3. Verify the installation

After installing, **place yourself in the project folder**, run the built-in self-test (this may create folders and files):

```bash
self-test
```

If everything is working correctly you will see the message **SurMod self-test PASSED** at the end of the output.

---

## Typical project layout

A standard SurMod project looks like this:

```text
my_project/
+-- configs.yaml          # experiment definitions
+-- data/
¦   +-- my_dataset.h5     # your HDF5 data files go here
+-- results/              # created automatically by training
¦   +-- ...
+-- main.py               # optional user script
```

- **`data/`**  place all your HDF5 datasets in this folder.  
  The configuration only needs the filename (e.g. `"my_dataset.h5"`); SurMod automatically looks for it under `data/`.
- **`configs.yaml`**  defines models, training settings and experiments.
- **`results/`**  training automatically writes checkpoints, metrics and plots here.
- **`main.py`** (optional)  a convenient place for your own scripts that call the SurMod API.

---

## Main API

SurMod exposes two primary high-level functions that cover the full lifecycle of an experiment.

### `train_model`

Trains a model (or runs hyperparameter tuning) for a named experiment defined in the YAML configuration.

```python
from surmod import train_model

# Standard supervised training
train_model(
    experiment="comp_full_gfr_baseline",
    config_path="configs.yaml",   # optional
    save_path=".",           # optional
)

# Hyperparameter tuning with Optuna
train_model(
    experiment="comp_full_gfr_baseline",
    tuner=True,
    time_hours=6.0,
)
```

**CLI equivalent:**

```bash
train-model --experiment comp_full_gfr_baseline
train-model --experiment comp_full_gfr_baseline --tuner --time-hours 6
```

## Inference with `predict`

Once a model has been trained, you can obtain predictions for new configurations with the high-level `predict` function.

```python
from surmod import predict

result = predict(
    model_path="results/my_dataset/baseline_GFR_Net_20260728_1430/GFR-Net.pth",
    X_values=[0.42, 1.15, 0.08],   # raw physical values, one per X_column
)

prediction = result["prediction"]   # shape (n_freq, n_out)  denormalised
axis       = result["axis"]         # physical frequency / sweep values
```

**Requirements**

- `model_path` must point to a `.pth` checkpoint produced by training.
- `X_values` must be a list or array whose length matches the number of geometry features the model was trained on (`X_columns` in the configuration).

The returned dictionary contains:

| Key                 | Content                                      |
|---------------------|----------------------------------------------|
| `prediction`        | Denormalised model output `(n_freq, n_out)`  |
| `axis`              | Physical values of the sweep variable        |
| `prediction_norm`   | Normalised prediction (optional use)         |

### `generate_results` (alias: `post_process`)

After training, generates metrics, plots (amplitude/phase, predictions vs true, training curves) and a human-readable `training_report.txt` from the run folder that contains the `.pth` checkpoint.

```python
from surmod import generate_results

generate_results("./results/{dataset}/{run}/")
```

**CLI equivalent:**

```bash
generate-results --path ./results/{dataset}/{run}
```

---

## Configuration System

All experiments are defined in a single YAML file (default: `configs.yaml`).

Example structure:

```yaml
experiments:
  my_experiment_name1:
    common:
      data_file: "dataset.h5"   # file must be located in the data/ folder
      save_name: "baseline"
      model_name: "GFR_Net"
      complex_format: "polar"
      n_out: 3
    model:
      hidden_dim: 512
      n_layers: 8
    dataset:
      X_columns: ["width", "height"]
      Y_columns: ["S_real", "S_imag"]
      axis_column: "freq_ghz"
      size: 3000               # nb of configurations used for training
    training:
      epochs: 1500
      batch_size: 512

 my_experiment_name2:
    common:
    ...
```

You can load a fully resolved config with:

```python
from surmod import load_config
config = load_config("my_experiment_name", config_path="configs.yaml")
```

---

## Configuration Parameter Reference

All parameters that the codebase currently reads are listed below, grouped by section. Defaults shown are those used when the key is omitted.

### `common`

| Parameter        | Type         | Default  | Description |
|------------------|--------------|----------|-------------|
| `model_name`     | str          | required | Model module to load |
| `data_file`      | str          | required | HDF5 filename located under the project `data/` folder |
| `save_name`      | str \| null  | `null`   | Prefix used when creating the results directory |
| `n_out`          | int          | `2`      | Number of output channels (2 = real/imag, 3 = polar) |
| `complex`        | bool         | `false`  | Whether the target is complex-valued |
| `complex_format` | str \| null  | `null`   | `"polar"` converts (real, imag) to (gain_dB, sin, cos) |
| `seed`           | int          | `42`     | Global random seed |
| `verbose`        | bool         | `true`   | Print detailed training progress |
| `cuda_nb`        | int          | `0`      | CUDA device index |
| `max_time`       | float        | `inf`    | Maximum training wall-clock time in seconds |

### `dataset`

| Parameter              | Type          | Default   | Description |
|------------------------|---------------|-----------|-------------|
| `X_columns`            | list[str]     | required  | Geometry / input feature column names |
| `Y_columns`            | list[str]     | required  | Target column names |
| `axis_column`          | str           | required  | Sweep variable (usually frequency) |
| `index_column`         | str           | `"index"` | Name of the unique row index column |
| `weights_type`         | str           | `"equal"` | Sample weighting strategy |
| `val_split`            | float         | `0.2`     | Fraction of geometries used for validation |
| `size`                 | int \| null   | `null`    | Max number of geometries for train+val |
| `column_filters`       | dict          | `{}`      | Keep only rows whose column values are in the given lists |
| `use_grouping`         | bool          | `false`   | Split train/val at the group level |
| `group_columns`        | list[str]     | `[]`      | Columns that define a group |

### `model`

| Parameter               | Type   | Default  | Description |
|-------------------------|--------|----------|-------------|
| `hidden_dim`            | int    | `512`    | Width of residual blocks |
| `n_fourier`             | int    | `64`     | Number of learnable Fourier features for the axis |
| `n_layers`              | int    | `8`      | Number of residual blocks in the main trunk |
| `n_out_layers`          | int    | `2`      | Number of residual blocks in the output head |
| `dropout`               | float  | `0.05`   | Dropout rate used during training |

### `training`

| Parameter            | Type   | Default         | Description |
|----------------------|--------|-----------------|-------------|
| `epochs`             | int    | `1500`          | Maximum number of training epochs |
| `patience`           | int    | = epochs        | Early-stopping patience |
| `patience_start`     | int    | `500`           | Epoch after which early-stopping starts |
| `batch_size`         | int    | `512`           | Mini-batch size |
| `lr`                 | float  | `0.0003`        | AdamW learning rate |
| `w_decay`            | float  | `1e-5`          | Weight decay |
| `metric`             | str    | `"weighted_mse"`| Loss metric |

### `tuning` (used only when `tuner=True`)

These override the default Optuna search spaces.

| Parameter     | Type        | Default search space                      | Description |
|---------------|-------------|-------------------------------------------|-------------|
| `hidden_dim`  | list[int]   | `[64, 128, 192, 256, 320, 384, 512]`      | Categorical |
| `n_fourier`   | list[int]   | `[16, 32, 48, 64, 96]`                    | Categorical |
| `n_layers`    | list[int]   | `[4, 10]` (min/max)                       | Integer range |
| `dropout`     | list[float] | `[0.0, 0.5]`                              | Float range |
| `lr`          | list[float] | `[1e-5, 1e-2]` (log-uniform)              | Float range |
| `w_decay`     | list[float] | `[1e-7, 1e-2]` (log-uniform)              | Float range |
| `batch_size`  | list[int]   | `[256, 512, 1024, 2048]`                  | Categorical |

---

## Data Format

SurMod reads data from a single **HDF5 file**.  
Think of an HDF5 file as a container that holds one big table of numbers.

### What the file must contain

Inside the file there must be one dataset (table) named **`features`**.

| Property | Requirement |
|----------|-------------|
| Name     | `features` |
| Shape    | (number of rows, number of columns) |
| Data type| 32-bit floating point numbers (`float32`) |

Each **row** of the table corresponds to one combination of:
- a geometry (design parameters), and
- one value of the sweep variable (usually a frequency).

### Required column information

The table must also store the names of its columns.  
This is done with an attribute called `column_names`.

Typical columns fall into four groups:

| Group          | Meaning                              | Examples |
|----------------|--------------------------------------|----------|
| X_columns      | Design parameters you can change     | `width`, `height`, `d1`  `d6` |
| axis_column    | The continuous variable (frequency)  | `freq_ghz` |
| Y_columns      | The measured or simulated output     | `S_real`, `S_imag` or `Gain` |
| Index          | A unique integer for every row       | `0, 1, 2, ` |

### Important rules (in plain language)

- Every geometry + frequency combination appears as its own row.
- The same geometry values are repeated for every frequency that belongs to that geometry.
- Complex numbers are stored as two separate columns (real part and imaginary part).
- The last column should be a simple integer index running from 0 to the number of rows - 1.
- All geometries should have the same number of frequency points.

### Minimal example of creating a data file

```python
import h5py
import numpy as np

# data is a 2-D array of shape (N_rows, N_columns)
# the last column must be the integer index
data = ... 

column_names = ["param_a", "param_b", "freq_ghz", "S_real", "S_imag", "index"]

with h5py.File("data/my_dataset.h5", "w") as f:
    dset = f.create_dataset(
        "features",
        data=data,
        compression="gzip",
    )
    dset.attrs["column_names"] = column_names
```

Place the resulting file inside the `data/` folder of your project.  
