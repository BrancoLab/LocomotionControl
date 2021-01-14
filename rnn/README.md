Code to facilitate the training and analysis of RNNs solving the controls problem

## Code organization
- analysis: all things related to analysing the dynamics of a trained network. helper functions, visualization etc. Analysis also includes `Pipeline` which can be used to carry out standardized analyses on trained networks.
- data: code to take the results of simulations using the 2WDD model in `control` and cleanup/process them to organize them in a dataset for training the RNN
- dataset: code to preprocess and copmile simulation data into a proper dataset for training with `pyrnn` and collection of different dataset types

`paths.py` has pointers to relevant paths for RNN work and `train_params.py` is used to set the params for training. 


## How to train
On **HPC** after `. control.sh`:
1. Update `Locomotion Control` with `update`
2. Update pyrnn with `interactive` followed by `updatedpyrnn`
3. `editrnn` to set the paramters
4. `train NAME` to start a trainin job with a given name
5. `q` to monitor training progress