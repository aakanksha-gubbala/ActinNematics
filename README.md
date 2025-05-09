## Phase field model for viscous inclusions in anisotropic networks


### Installation

Simply add the `ModelC_Cubic.py` to your project directory

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
mc.N = 512		# number of grid points
mc.L = 500		# side length of square box
mc.DT = 1e-6	# time step
mc.TIME = 100	# final simulation time
mc.TOL = 1e-4	# error tolerance
mc.SKIPS = 100	# save solution at every 100 timesteps

# Constants that appear in Eqs. 9 & 10 of main text
mc.BETA = 1.0 
mc.GAMMA = 0.1
mc.CHI = 1.0
mc.E1 = 1.0
mc.E3 = 1.0
mc.PHI_AVG = 0.30

# Start the simulation
mc.run()

# Export model parameters to .txt file
mc.export_params()
```

### MIT License

Copyright (c) 2025 Aakanksha Gubbala.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.