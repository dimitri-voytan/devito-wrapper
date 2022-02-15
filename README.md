# Devito Wrapper
Wrapper for Devito to ease computation of elastic, acoustic, and viscoelastic wave fields. Organize experiments in config files and save state for future reference.

Currently only 2D elastic computation is supported and most features are not fully developed nor tested.

# Installation

The primary requirement is [devito](https://github.com/devitocodes/devito) which handles the discretization of the domain and the solver.
More requirements updates and installation instructions coming soon.

# Quickstart :rocket:

An example is setup with a small verison of the Marmousi model. 

Assuming you have a working devito environment setup run

```
cd elastic
python3 elastic_run.py
```

This will write a shot gather to disk in a numpy object array which contains the source coordinate, reciever coordinates,
<img src="https://render.githubusercontent.com/render/math?math=\nabla \cdot \mathbf{v}"> (rec1), and <img src="https://render.githubusercontent.com/render/math?math=\tau_{zz}"> (rec2). 

More coming soon!
