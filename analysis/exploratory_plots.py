import sys
from tpd import recorder
import matplotlib.pyplot as plt

sys.path.append('./')
from pathlib import Path

from fcutils.plot.figure import clean_axes

from data.dbase import db_tables
from data.dbase.hairpin_trace import HairpinTrace
from analysis.visuals import plot_tracking_xy

base_folder = Path(r'D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis')

'''
    Plot a few exploratory plots for each session a recording was performed on.

        1. Overview of tracking data  
'''


hairpin = HairpinTrace()
hairpin.build()

sessions = (db_tables.ValidatedSession * db_tables.Session & 'is_recording=1' & 'arena="hairpin"')

for session in sessions:
    # start a new recorder sesssion
    recorder.start(base_folder=base_folder, name=session['name'], timestamp=False)

    # load tracking data
    tracking = db_tables.Tracking.get_session_tracking(session['name'], body_only=False)
    body_tracking = tracking.loc[tracking.bpname == 'body'].iloc[0]

    # assign points to position on linearized track
    body_tracking['arena_segment'] = hairpin.assign_tracking(body_tracking.x, body_tracking.y)

    # plt tracking data
    f, axes = plt.subplots(ncols=3, figsize=(18, 8))
    f.suptitle(session['name'])
    f._save_name = 'tracking_data_2d'
    for n, key in enumerate(('arena_segment', 'direction_of_movement', 'orientation')):
        # plot_tracking_xy(body_tracking, key=key, ax=axes[n], skip_frames=5, cmap='Pastel1', alpha=.5)
        # axes[n].set(title=f'clrd by "{key}"')

        hairpin.draw(axes[n], body_tracking)
        break

    clean_axes(f)

    plt.show()
    break

    recorder.add_figures(svg=False)
    plt.close('all')