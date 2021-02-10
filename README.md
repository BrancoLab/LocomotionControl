Code organization. Each subfolder has a dedicated README with additional info.

## control
Code for the simulation and control of two wheel differential drive robot. It include codes to generate or load trajectories that the robot has to follow, to model the robot's physics and to control the robot so that it does indeed follow the trajectory.


## data_wrangling
Code used to take data from other lab memebers and put it into a format that can be used for generating data for RNN training. It  also takes care of taking tracking data and producing trajectories for 2WDD simulation and cleaning simulation results.

## experiment_validation
Code for validating models' predictions against experimental data. Currently this deal with tracking/cleaning up/loading experimental data and checking the prediction of a 2WDD  model run on trajectories from experimental data

## kinematics
Code to analyze the kinematics of mouse locomotion (e.g. extract steps info)

## RNN
Code to train and analyze RNNs. The training uses the datasets specfied in rnn.dataset which in turn create datasets from the results of simulation data. analysis includes code to visualize and analyze RNN dynamics.