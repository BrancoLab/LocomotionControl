
import sys
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Union

sys.path.append("./")

from tpd import recorder


from data.dbase.db_tables import Probe, Unit, Recording, Tracking, LocomotionBouts, Session, Behavior, FiringRate
from data import data_utils
from analysis.exploratory_plots import plot_n_units_per_channel, plot_hairpin_tracking,  plot_unit, plot_unit_firing_rate
'''
    Inspects the data from one recording session. Can plot:
        - tracking data
        - probe channels data
        - single units activity
'''


TARGETS = (
    "PRNr",
    "PRNc",
    "CUN",
    "GRN",
    "MB",
    "PPN",
    "RSPagl1",
    "RSPagl2/3",
    "RSPagl5",
    "RSPagl6",
    "RSPd1",
    "RSPd2",
)



class Inspector:
    base_folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis")

    def __init__(self, session_name:str, events_window:int=3*60, firing_rate_window:int=100):
        self.session_name = session_name
        self.hairpin = Session.on_hairpin(self.session_name)

        self.events_window = events_window
        self.firing_rate_window = firing_rate_window

        # start logging
        recorder.start(base_folder=self.base_folder, name=session_name, timestamp=False)

        # get session data
        self.fetch()


    def fetch(self):
        # get tracking data
        logger.info('Fetching tracking')
        self.tracking =  Tracking.get_session_tracking(
            self.session_name, body_only=True
        )
        self.downsampled_tracking = Tracking.get_session_tracking(
            self.session_name, body_only=True
        )
        data_utils.downsample_tracking_data(self.downsampled_tracking, factor=10)

        # get locomotion bouts
        logger.info('Fetching locomotion bouts')
        self.bouts = LocomotionBouts.get_session_bouts(self.session_name)
        self.out_bouts = self.bouts.loc[self.bouts.direction=='outbound']
        self.in_bouts = self.bouts.loc[self.bouts.direction=='inbound']
        self.out_bouts_stacked = data_utils.get_bouts_tracking_stacked(self.tracking, self.out_bouts)
        self.in_bouts_stacked = data_utils.get_bouts_tracking_stacked(self.tracking, self.in_bouts)

        # get units and recording sites
        recording = (Recording & f'name="{self.session_name}"').fetch()
        logger.info('Fetching ephys data')
        self.units = pd.DataFrame((Unit * Unit.Spikes * FiringRate & recording & f'firing_rate_std={self.firing_rate_window}').fetch())
        self.rsites = pd.DataFrame((Probe.RecordingSite & recording).fetch())
        logger.debug(f'Found {len(self.units)} units')

        # get tone onsets
        self.tone_onsets = (Behavior * Behavior.Tones & f'name="{self.session_name}"').fetch1('tone_onsets')

    def plot(self,
        tracking:bool = False,
        probe:bool = False,
        unit:Union[bool, str, int] = None,  # False, ID of unit to show or 'all'
        firing_rate:bool=False,
        show:bool=True,
    ):

        if tracking:
            logger.info('Plotting TRACKING')
            if self.hairpin:
                plot_hairpin_tracking(
                        self.session_name,
                        self.tracking,
                        self.downsampled_tracking,
                        self.bouts,
                        self.out_bouts,
                        self.in_bouts,
                        self.out_bouts_stacked,
                        self.in_bouts_stacked,
                        )
            else:
                raise NotImplementedError('Adjust open arena plotting code')

        if probe:
            logger.info('Plotting PROBE')
            plot_n_units_per_channel(self.session_name, self.units, self.rsites, TARGETS)

        if unit is not None:
            logger.info(f'Plotting UNIT {unit}')
            plot_unit(
                self.session_name,
                self.tracking,
                self.bouts,
                self.out_bouts,
                self.in_bouts,
                self.units,
                unit,
                self.tone_onsets,
                WINDOW = self.events_window
            )

            if plot_unit_firing_rate:
                if not isinstance(unit, int):
                    raise ValueError(f'Plotting firing rate only works for a single unit, cant plot it for: {unit}')
                plot_unit_firing_rate(self.units.loc[self.units.unit_id == unit])

        if show:
            plt.show()

        recorder.add_figures(svg=False)
        plt.close("all")


if __name__ == '__main__':
    insp = Inspector('FC_210714_AAA1110750_r4_hairpin')
    insp.plot(
        tracking = False,
        probe = False,
        unit = 1237,
        show=True
    )