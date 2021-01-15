Code to facilitate the training and analysis of RNNs solving the controls problem

## Code organization
- analysis: all things related to analysing the dynamics of a trained network. helper functions, visualization etc. Analysis also includes `Pipeline` which can be used to carry out standardized analyses on trained networks.
- data: code to take the results of simulations using the 2WDD model in `control` and cleanup/process them to organize them in a dataset for training the RNN
- dataset: code to preprocess and copmile simulation data into a proper dataset for training with `pyrnn` and collection of different dataset types

`paths.py` has pointers to relevant paths for RNN work and `train_params.py` is used to set the params for training. 

## How to prepare data for RNN training
The datasets classes allow you to take the results of running the 2WDD simulations and organize the data in a format that can be used for training RNNs

**note:** before creating the datasets, you'll want to manually inspect the simulations' outputs to remove runs in which the simulation failed (e.g. control failure). For that you can use `rnn.data.discard_trials_manually.py`

After that, you can create a DataSet class in `rnn.dataset.dataset`, these will be called something like `PredictTauFromXYT` and should subclass the `Dataset` and `Preprocessing` classes. Dataset calsses should have a name and names for inputs and outputs and should have  `get_inputs` and `get_outputs` methods.

### `Preprocessing`
The `Preprocessing` class loads the results of the 2WDD simulations and prepares them for use in a dataset (e.g. scaling, normalization, train/test splitting).  This is where `get_inputs` and `get_outputs` play an important role. After the data have been processed `Dataset` feeds them to the RNN but it doesn't mody the data anymore

### `Dataset`
This class takes the data that have been cleaned up by `Preprocessing` according to the methods specified in a given datase class and implemetns methods for:
* augmenting the data
* splitting trials into chunks
* adding warmup to trials
* return batches of data that can be used for training the RNNs



## How to train on HPC
On **HPC** after `. control.sh`:
1. Update `Locomotion Control` with `update`
2. Update pyrnn with `interactive` followed by `updatedpyrnn`
3. `editrnn` to set the paramters
4. `train NAME` to start a trainin job with a given name
5. `q` to monitor training progress
