# OPTTI: Optimization of Temporal Interference

## Installation

1. Ensure you have python 3.9.x installed and activated, which is the version officially supported by simnibs.
2. Clone this repository.
3. Navigate to the repository and install `optti` using `pip install .`.
4. Begin using optti.

You can also install simnibs (https://simnibs.github.io/simnibs/build/html/index.html) using their installer. However, you need to activate the simnibs environment before installing optti.

## Usage

**Initialization:**

```python
import optti

model_dir = 'path/to/simnibs/subject/folder'
current = 0.001 # in A
pre_calculation_dir = 'path/to/all/pre-calculated/simulations'
eeg_coord_sys = '10-10' # EEG system

# Initialize an optimization project:
opt = optti.OptTI(model_dir, pre_calculation_dir, eeg_coord_sys=eeg_coord_sys, current=current) 

# Before running TI optimization, pre-calculate all the tDCS simulations N-1 times, where N is the number of electrodes.
opt.pre_calculate(cores=8)

# Align mesh files from the pre-calculated simulations so they all share the same identical point coordinates.
opt.align_mesh()
```

**Compute TI density**

```python
opt.get_TI_density(('AF7', 'O1'),('F8', 'PO8'), save_TI_mesh=True)
```

**Compute TI focality**

```python
opt.get_TI_focality(
    ('AF7', 'O1'),
    ('F8', 'PO8'),
    (0.7879,  21.1037, -22.0770), # target center, we also support using a list of coordinates for multi-target focality
    2, # target radius, we also support using a list of radius for multi-target focality.
    )
```

**Discrete Optimization**

Under construction