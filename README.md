# All of my code

Rough organisation:
- /src/ contains the code
- `util.py` contains generally useful functions
- `server_utils.py` contains functions and classes for interfacing with the cluster
- `pt_utils.pt` contains utilities for pytorch
- `bae.py` contains the Binary Autoencoders
- `students.py` contains classes for various PyTorch modules
- `super_experiments.py` and `experiments.py` contain classes for organizing my experiments
- `plotting.py` contains functions for plotting
- `anime.py` contains animation utilities

And other files are more idiosyncratic. 

An example usage of the code is contained in `/scripts/cifar_interventions.py` which trains a convnet on cifar10 or mnist and then applies one of my matrix factorization methods to the hidden representations.