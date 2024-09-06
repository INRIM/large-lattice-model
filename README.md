# large-lattice-model
A python package to model the motional spectra of atoms trapped in a one-dimensional optical lattice.


# Requirements
Large-lattice-model requires the calculation of Mathieu special function.
The current `scipy` implementation of Mathieu functions [is bugged.](https://github.com/scipy/scipy/pull/14577)
It is reccomended to install the [GSL Library](https://www.gnu.org/software/gsl/), for example using:

`$ sudo apt install libgsl-dev`

Large-lattice-model will use `ctypes` to import the GSL implementation and will fallback to `scipy` if it does not find GSL.