# OPTTI: Optimization of TI

## Installation

Install simnibs (https://simnibs.github.io/simnibs/build/html/index.html) either using installer or pip

```bash
# Notice that you need to install the correct version of simnibs for your python version
pip install https://github.com/simnibs/simnibs/releases/download/v4.0.0/simnibs-4.0.0-cp39-cp39-win_amd64.whl
```

**Activate the simnibs environment (optional)**
If you install simnibs using installer, you need to activate the simnibs environment

```bash
conda activate /path/to//SimNIBS/simnibs_env
```

Install optti

```bash
pip install optti

```

## Usage

Run pre-calculations

```python
import optti

opt = optti.Optti('ernie', 'path/to/results/folder')

```

Run simulation pruning to reduce the storage size (optional)

```python
opt.prune_storage()
```

Simulate a specific electrode configuration

```python
opt.simulate(e1, e2, geometry, current)
```

Run optimization

```python
opt.run((x,y, z), r, current)
```
