using Plots


using jcontrol
using jcontrol.visuals


trials = load_cached_trials(; keep_n = 300,)


plt = draw(:arena)
draw!.(trials; lw=1, alpha=.2)    
display(plt)