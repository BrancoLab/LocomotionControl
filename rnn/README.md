Code to facilitate the training and analysis of RNNs solving the controls problem

## Code organization
- analysis: all things related to analysing the dynamics of a trained network. helper functions, visualization etc.
- data: code to take the results of simulations using the 2WDD model in `control` and cleanup/process them to organize them in a dataset for training the RNN
- dataset: code to preprocess and copmile simulation data into a proper dataset for training with `pyrnn` and collection of different dataset types

`paths.py` has pointers to relevant paths for RNN work and `train_params.py` is used to set the params for training. 