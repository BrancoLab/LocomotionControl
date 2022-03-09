## Usage
cmd+shift+P to open the commands palette, start julia REPL.

If you're not in `analysis/behavior`, in bash mode:
```
cd analysis/behavior
```

in PKG mode:
```
(@v1.7) pkg> activate .
  Activating project at `~/Documents/Github/LocomotionControl/analysis/behavior`

(jcontrol) pkg> 
```

(You might have to install all dependencies, I'm not sure if does that automatically).


### Do MTM
Then, to run the whole thing (control problem + forward integration + plotting), execute `do_mtm.jl`

In it you'll see you have options to run each part separately:
```Julia
DO_MODEL    = false
DO_FORWWARD = true
```
The first time you run the script you must run both, but if you just want to do the ODE or just the plotting you can skip unnecessary steps. 

## Description
In `src`, `bike.jl` defines a `Bicycle` type which stores info about the bike's geometry and for plotting it and `State` type which can be used to represent the bike's state at a moemnt in time (used to define initial conditions in control problem).

`track.jl` defines the `Track` type which stores track parameters, width a constructor method that loads the track's waypoint coordinates from a numpy file and generates the track. You can use `keep_n_waypoints` to only keep the first N waypoints and produce a shorter track. The `Track` constructor takes the X, Y coordinates of the waypoints and interpolates to a higher resolution (with `resolution` parameter setting that). Then it computes things like `s` (curvilinear distance), $\theta$ for the orientation, curvature etc. It also has a function $\kappa$ (\kappa, not k) which gives the curvature as a function of `s`, that's used to solve the control model.


The optimal control problem with InfiniteOpt is handled by `contro.jl`. This defines the `ControlOptions` type to set options for the control model (IPoPT parameters or things like variables bounds). The way the optimal control problem is setup and solved is the same as what we discussed earlier, but for the bicycle kinematics. `create_and_solve_control` accepts to `State` arguments to set the initial and final conditions.


The forward integration is done by `forward_model.jl`'s `run_forward_model` function. Inside this are defined the `control` and `kinematics!` funcitions used by `DifferentialEquations` to solve the ODEs. `controls` is the critical one, currently the two options (partially commented out) are zeroth-order and first-order (broken) interpolation. It returns a `Solution` type that keeps stores each variable's value at each $\delta t$.


Finally `visuals.jl` has the plotting code. 
```Julia
function summary_plot(model::InfiniteModel, wrt::Symbol)
```
plots the solution to the control problem while 
```Julia
function summary_plot(model::Solution, controlmodel::InfiniteModel,  track::Track, bike::Bicycle)
```
plots the solution to the ODE integration + some of those for the control model for reference. It also plots the bike. I haven't gotten around to putting axes labels and titles, but you should get it from the code. 

Cheers,
F.