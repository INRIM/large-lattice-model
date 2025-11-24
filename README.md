# large-lattice-model
A python package to model the motional spectra of atoms trapped in a one-dimensional optical lattice.


## Requirements
Large-lattice-model requires the calculation of Mathieu special function.
The current `scipy` implementation of Mathieu functions [is not continuous.](https://github.com/scipy/scipy/pull/14577)
It is reccomended to install the [GSL Library](https://www.gnu.org/software/gsl/), for example using:

    sudo apt install libgsl-dev

Large-lattice-model will use `ctypes` to import the GSL implementation and will fallback to `scipy` if it does not find GSL.
Previous versions of the package used [`pygsl`](https://github.com/pygsl/pygsl), but the current version of `pygsl` does not implement [special functions](https://github.com/pygsl/pygsl/issues/55).



## License

[MIT](https://opensource.org/licenses/MIT)

## References

This package implements many ideas from the work of NIST Yb optical lattice clock group:

[Beloy et al., Phys. Rev. A, 101, 053416 (2020).](https://doi.org/10.1103/PhysRevA.101.053416)

It is developed for use in the INRIM Yb optical lattice clock IT-Yb1, see for example:


[Goti et al., Metrologia, 60, 035002 (2023).](https://dx.doi.org/10.1088/1681-7575/accbc5)

## Acknowledgments
This work has received funding from the European Partnership on Metrology, co-financed by the European Union’s Horizon Europe Research and Innovation Programme and by the Participating States, under grant number 22IEM01 TOCK.

![badge](./docs/source/Acknowledgement%20badge.png)

## Authors

(c) 2021-2024 Marco Pizzocaro - Istituto Nazionale di Ricerca Metrologica (INRIM)


## Basic usage
Install from github, e.g.:

    pip install git+https://github.com/INRIM/large-lattice-model.git

Import the package as:

    import large_lattice_model as llm

By default, Large-lattice-model is set for ytterbium-171 properties. To work with Sr, call

    llm.set_atom('87Sr')

Example plot:

    freq = np.linspace(0, 50e3, 100)
    sideband = llm.sidebands(freq, D=100, Tz=1e-6,Tr=1e-6, r=1, b=1, w=1e3)
    plt.figure()
    plt.plot(freq, sideband)

