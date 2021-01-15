Code for the 2WDD simulation of a locomoting mice + control solution for having the 2WDD robot follow given trajectory. 
It includes code to generate artificial trajectories, live plot and summary of the simulations etc.

## To run 2WDD on simulated data
The code to generate simulated trajectories is in `_world.py`
`./winstor.py` is what takes care of running the simulation on winstor by first **go to `config.py` and make sure that the trajectory type is set to simulated**.

Once all settings are OK and you are on HPC, you can run: `launch.sh N` where N is the number of simulations you want.

## To run 2WDD on real data
**note:** you'll want to manually inspect behavioural data before to remove e.g. incomplete trials *before* running the simulations. For that use `./clean_trials_for_2wdd.py`.

The code that cleans up data and creates trajectories for the simultion is in `_world.py`.
`./winstor.py` is what takes care of running the simulation on winstor by first **go to `config.py` and make sure that the trajectory type is set to tracking**.

To run all files in parallel, you'll need to generate a `.sh` file for each trial and then run `launch_all_trials.sh`. You can generate the trials files with: `rnn\data\generate_bash_scripts.py`. To speify where the tracking data ar saved use: `control\paths.py > trials_cache` and `> winstor_trial_cache`. 


## bash files and what they do
* `control_hpc.sh` just runs `./winstor.py` with whatever settings are specified
* `launch_all_trials.sh` runs a `./winstor.py` for each individual trial in the dataset (each trial should have a .sh file saved in `trials_bash_files`)
* `launch.sh` allows you to easily lunch N simulations by invoking `control_hpc.sh` N times.