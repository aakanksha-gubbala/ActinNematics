## Phase field model for viscous inclusions in anisotropic networks

This repository contains the main simulation code for the following article:

> Gubbala, A., Jena, A. M., Arnold, D. P., & Takatori, S. C. Phase field model for viscous inclusions in anisotropic networks. _Soft Matter_ (2025). DOI: [10.1039/D5SM00478K](https://doi.org/10.1039/D5SM00478K).

### Installation

Simply add the `ModelC_Cubic.py` to your project directory.

### Dependencies

This script requires:

* `numpy` (>= 2.0.0)
* `h5py` (>= 3.11.0)
* `cupy` (>= 12.0) for GPU acceleration (optional)

### Example utilization

Below, you'll find a demonstration on how to utilize the `ModelC_2D` class:

```
mc = ModelC_2D(<seed for random number generator>, 
			   <path to save file>, 
			   device='gpu' or 'cpu')
# Numerical parameters
mc.N     = 512	# number of grid points
mc.L     = 500  # side length of square box
mc.DT    = 1e-6 # time step
mc.TIME  = 100  # final simulation time
mc.TOL   = 1e-4 # error tolerance
mc.SKIPS = 100  # save solution at every 100 timesteps

# Constants that appear in Eqs. 9 & 10 of main text
mc.BETA    = 1.0 
mc.GAMMA   = 0.1
mc.CHI     = 1.0
mc.E1      = 1.0
mc.E3      = 1.0
mc.PHI_AVG = 0.30

# Start the simulation
mc.run()

# Export model parameters to .txt file
mc.export_params()
```
