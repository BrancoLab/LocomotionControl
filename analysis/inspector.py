import sys
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Union

sys.path.append("./")

from tpd import recorder

from data.dbase.db_tables import (
    Unit,
    Recording,
    Tracking,
    LocomotionBouts,
    Session,
    Probe,
    Tones,
)
from data import data_utils
from analysis.exploratory_plots import (
    plot_n_units_per_channel,
    plot_hairpin_tracking,
    plot_unit,
    plot_unit_firing_rate,
    render_probe_3d,
    render_probe_regions_slices,
)

"""
    Inspects the data from one recording session. Can plot:
        - tracking data
        - probe channels data
        - single units activity
"""


TARGETS = (
    "PRNr",
    "PRNc",
    "CUN",
    "GRN",
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

    def __init__(
        self,
        session_name: str,
        events_window_s: int = 3,
        firing_rate_window: int = 100,
    ):
        self.session_name = session_name
        self.hairpin = Session.on_hairpin(self.session_name)

        self.events_window = events_window_s * 60
        self.firing_rate_window = firing_rate_window

        # start logging
        recorder.start(
            base_folder=self.base_folder, name=session_name, timestamp=False
        )

        # get session data
        self.fetch()

    def fetch(self):
        # get tracking data
        logger.info("Fetching tracking")
        self.tracking = Tracking.get_session_tracking(
            self.session_name, body_only=True
        )
        self.downsampled_tracking = Tracking.get_session_tracking(
            self.session_name, body_only=True
        )
        data_utils.downsample_tracking_data(
            self.downsampled_tracking, factor=10
        )

        # get when the tone is one
        self.tracking["tone_on"] = Tones.get_session_tone_on(self.session_name)

        # get locomotion bouts
        logger.info("Fetching locomotion bouts")
        self.bouts = LocomotionBouts.get_session_bouts(self.session_name)
        logger.info(f'Found {len(self.bouts)} bouts')
        self.out_bouts = self.bouts.loc[self.bouts.direction == "outbound"]
        self.in_bouts = self.bouts.loc[self.bouts.direction == "inbound"]

        self.bouts_stacked = data_utils.get_bouts_tracking_stacked(
            self.tracking, self.bouts
        )
        self.out_bouts_stacked = data_utils.get_bouts_tracking_stacked(
            self.tracking, self.out_bouts
        )
        self.in_bouts_stacked = data_utils.get_bouts_tracking_stacked(
            self.tracking, self.in_bouts
        )

        # get units and recording sites
        recording = (Recording & f'name="{self.session_name}"').fetch(as_dict=True)[0]
        cf = recording['recording_probe_configuration']
        logger.info("Fetching ephys data")
        self.units = Unit.get_session_units(
            self.session_name,
            cf,
            spikes=True,
            firing_rate=True,
            frate_window=self.firing_rate_window,
        )
        self.rsites = pd.DataFrame((Probe.RecordingSite & recording & f'probe_configuration="{cf}"').fetch())
        logger.debug(f"Found {len(self.units)} units")

        ids = "\n".join(
            [
                f'unit: {unit.unit_id} in "{unit.brain_region}"'
                for i, unit in self.units.iterrows()
            ]
        )
        logger.debug(f"Available units: \n{ids}")

        # get tone onsets
        self.tone_onsets = (Tones & f'name="{self.session_name}"').fetch1(
            "tone_onsets"
        )

    def plot(
        self,
        tracking: bool = False,
        probe: bool = False,
        unit: Union[
            bool, str, int
        ] = None,  # False, ID of unit to show or 'all'
        firing_rate: bool = False,
        show: bool = True,
    ):
        logger.warning('Inspector should plot firing rate vs heading direction and heading ang vel not ang vel')
        
        # tracking
        if tracking:
            logger.info("Plotting TRACKING")
            if self.hairpin:
                plot_hairpin_tracking(
                    self.session_name,
                    self.tracking,
                    self.downsampled_tracking,
                    self.bouts,
                    self.out_bouts,
                    self.in_bouts,
                    self.bouts_stacked,
                    self.out_bouts_stacked,
                    self.in_bouts_stacked,
                )
            else:
                raise NotImplementedError("Adjust open arena plotting code")

        # probe
        if probe:
            logger.info("Plotting PROBE")
            (self.base_folder / 'probe').mkdir(exist_ok=True)
            plot_n_units_per_channel(
                self.session_name, self.units, self.rsites, TARGETS
            )
            # render_probe_3d(
            #     self.rsites, self.base_folder / 'probe' / self.session_name, TARGETS
            # )
            # render_probe_regions_slices(
            #     self.rsites, self.base_folder / 'probe' / self.session_name, TARGETS
            # )

        # unit ephys
        if unit is not None:
            logger.info(f"Plotting UNIT {unit}")
            plot_unit(
                self.tracking.mouse_id,
                self.session_name,
                self.tracking,
                self.bouts,
                self.out_bouts,
                self.in_bouts,
                self.bouts_stacked,
                self.units,
                unit,
                self.tone_onsets,
                WINDOW=self.events_window,
            )

            # firing rate
            if firing_rate:
                logger.info("Plotting firing rate")
                if not isinstance(unit, int):
                    raise ValueError(
                        f"Plotting firing rate only works for a single unit, cant plot it for: {unit}"
                    )
                plot_unit_firing_rate(
                    self.units.loc[self.units.unit_id == unit].iloc[0]
                )

        # save figs
        recorder.add_figures(svg=False)

        # show figs and close
        if show:
            logger.debug("Showing plot")
            plt.show()

        plt.close("all")


if __name__ == "__main__":
    for rec in Recording().fetch("name"):
        if "open" in rec:
            continue
        
        try:
            insp = Inspector(
                rec,
                firing_rate_window=33,
                events_window_s=5,
            )

            insp.plot(
                tracking=False, probe=True, unit=None, firing_rate=False, show=True,
            )

            # breakq

        except Exception as e:
            # logger.warning(e)
            raise ValueError(e)
