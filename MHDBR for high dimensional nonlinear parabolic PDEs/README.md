# MHDBR-FBSDE

This repository provides the PyTorch implementation of the multistep
high-order deep backward regression (MHDBR) methods for solving
forward-backward stochastic differential equations and the associated
parabolic partial differential equations.

The code includes the MHDBR schemes with (J=1,2,3,4), refined Euler
approximations for the starting values of the multistep schemes, and the
MDBDP method used for numerical comparison.  Neural-network regression and
conditional Monte Carlo sampling are implemented with PyTorch.

## Requirements

- Python 3.10 or later
- PyTorch
- NumPy

Install the required packages with

```bash
pip install -r requirements.txt
```

Each program can be run from the repository root.  Use `--help` to view its
available parameters, for example:

```bash
python examples/example1/run_j1.py --help
python examples/example1/run_mdbdp.py --help
```

If you use this code, please cite the accompanying MHDBR paper.
