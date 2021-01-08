Code to preprocess and analyse data from experiments aimed at validating the modelling work. 

## Tracking - gait analysis
In the first round of experiments we've filed mice from below as they escaped to the shelter in an arena that forced them to take many turns. 
The goal is to analyse their locomotion kinematics to compare them with the 2WDD robot. 

The data are videos + timestamps and stim times acquired with Bonsai (video at 60fps).
Raw data are saved on winstor at "Z:\swc\branco\Federico\Locomotion\control\experimental_validation\2WDD_raw"

The analysis steps are:
1. check that the files are fine (e.g. no dropped frames)
2. extract clips from around the stimuli
3. run DLC on these clips 
4. analyze DLC data to extract steps times
5. analyse steps data to inspect kinematics

After that, to validate the 2WDD model
1. run the 2WDD on the trajectories from these experiments
2. compare