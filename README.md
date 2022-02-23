# Devito Wrapper
Wrapper for Devito to ease computation of elastic, acoustic, and acoustic anisotropic wave fields. Organize experiments in config files and save state for future reference.

Currently only 2D isotropic elastic and constant density acoustic computation is supported and most features are neither fully developed nor tested.

The solvers are based on the [devito seismic examples](https://github.com/devitocodes/devito/tree/master/examples/seismic) with improvements (soon) to ease the development of wave propagation experiments in realistic media. 

# Installation :gear:

The primary requirement is [devito](https://github.com/devitocodes/devito) which handles the discretization of the domain and the solver. It is recommended to follow the installation instructions on that repo.

More requirements updates and installation instructions coming soon.

# Quickstart :rocket:

An example is setup with a small verison of the Marmousi model. 

Assuming you have a working devito environment setup run

```
cd acoustic
python3 acoustic_run.py
```

This will write 20 shot gathers to disk in numpy object arrays which contains the source coordinate, reciever coordinates, and pressure data. 

To run with a different velocity model, acquisition geometry, source frequency, etc. edit the config file `config.yaml` accordingly.
Some of the parameters in the config file are not yet implemented; these are noted as such. 

Some guidelines for running:

1) The velocity model and other physical parameters should be stored in numpy arrays with shape `(nx, nz)`
2) The `nav.pkl` controls the source and reciever positions. Check the script `prepare_geometry.py` to see how to prepare this file. Currently, simultaneous source acquisition is not supported.

# Technical Notes

The damping absorbing boundaries provided in the devito examples are unsatisfactory for many applications. I am working on rewriting the devito examples with PML absorbing boundaries. In the meantime, the velocity model is padded based on the two way travel time of straight rays from the source location to the model boundary so that the numeric reflections are largely removed. This is reasonably affordable in 2D. Note that the padding is not robust if e.g. there are high velocity regions in the shallow subsurface, so use a few damping layers as well. 

Currently the solver is limited to 2nd order in time. I will try to include higher order time discretizations, improved spatial stencils, etc. as time permits. 

# Coming Soon

1. Parallelization over source location via dask :white_check_mark:
2. Complete acoustic :white_check_mark: and acoustic tti cases :white_check_mark:
3. Unfinished config features
4. PML absorbing boundary conditions
5. Extension to 3D
6. GPU support
7. Simultaneous sources
