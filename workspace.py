from proj.plotting.results import plot_results
import matplotlib.pyplot as plt
import logging

logging.disable(logging.DEBUG)
log = logging.getLogger("rich")

fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/control/tracking_200920_220443_6689/results"


plot_results(fld)
plt.show()
