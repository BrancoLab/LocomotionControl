
import sys
sys.path.append('./')

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from pathlib import Path

from tpd import recorder

import draw
from data.dbase.db_tables import ROICrossing
from data import arena, colors

from analysis._visuals import move_figure
from analysis import visuals

folder = Path(r'D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\behavior')
recorder.start(
    base_folder=folder, name='roi_crossings', timestamp=False
)

for ROI in arena.ROIs_dict.keys():
    # create figure and draw ROI image
    f = plt.figure(figsize=(20, 12))
    f.suptitle(ROI)
    f._save_name=f'roi_{ROI}'
    axes = f.subplot_mosaic(
        """
        AABC
        AADE
        FFGH
        """
    )

    for ax in "ABCDE":
        draw.ROI(ROI, ax=axes[ax], set_ax=True if ax == 'A' else False)


    # fetch from database
    crossings = pd.DataFrame((ROICrossing & f'roi="{ROI}"' & 'mouse_exits=1').fetch())
    logger.info(f'Loaded {len(crossings)} crossings')

    color = arena.ROIs_dict[crossings.iloc[0].roi].color

    # draw tracking
    for i, cross in crossings.iterrows():
        if i % 30 == 0:
            axes['A'].scatter(cross.x[0], cross.y[0], zorder=100, color=color, lw=.5, ec=[.3, .3, .3])
            draw.Tracking(cross.x, cross.y, ax=axes['A'], alpha=.7)

    # draw heatmaps for speed and acceleration
    visuals.plot_heatmap_2d(crossings, gridsize=25, key='speed', ax=axes['B'],)
    visuals.plot_heatmap_2d(crossings, gridsize=25, key='acceleration', ax=axes['D'], vmin=-2.5, vmax=2.5, cmap='bwr')
    visuals.plot_heatmap_2d(crossings, gridsize=25, key='thetadot', ax=axes['C'], vmin=-600, vmax=600, cmap='bwr')
    visuals.plot_heatmap_2d(crossings, gridsize=25, key='thetadotdot', ax=axes['E'], vmin=-25, vmax=25, cmap='bwr')


    # draw histogram of duration
    axes['F'].hist(crossings.duration, bins=50, color=color)

    # clean up
    axes['A'].set(ylabel=ROI)
    axes['F'].set(xlim=[0, 5], xlabel='duration (s)')
    axes['B'].set(title='speed', xticks=[], yticks=[])
    axes['D'].set(title='acceleration', xticks=[], yticks=[])
    axes['C'].set(title='ang. vel.', xticks=[], yticks=[])
    axes['E'].set(title='ang. acc.', xticks=[], yticks=[])


    f.tight_layout()
    move_figure(f, 50, 50)

    recorder.add_data(crossings, f'{ROI}_crossings', fmt='h5')

    # break
recorder.add_figures(svg=False)
recorder.describe()
plt.show()
