import numpy as np
from pathlib import Path
import pandas as pd
from typing import Union
from loguru import logger

from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from myterial.utils import rgb2hex


def select_rows(
    data: Union[pd.DataFrame, np.array],
    skip: int = 200,
    pitch: int = 20,
    N: int = 384,
):
    """Selects rows of dataframe/array corresponding to electrodes position"""
    # skip probe tip
    data = data[skip:]
    data = data[::pitch]
    data = data[:N]
    return data


def place_probe_recording_sites(
    probe_metadata: dict, n_sites: int = 384
) -> list:
    # get brainglobe atlas
    atlas = BrainGlobeAtlas("allen_mouse_25um")

    # load points (1 every um of inserted probe)
    try:
        points = np.load(probe_metadata["reconstructed_track_filepath"])
    except TypeError:
        logger.warning("Could not load reconstructed track file")
        return

    if not probe_metadata["implanted_depth"] * 1000 == len(points):
        logger.warning(
            "The reconstructed probe should have one point for every um of inserted probe"
        )
        return

    # discard first 200um because there's no electrodes threshold
    # one electrode every 20um
    # keep 384 electrodes
    electrodes = select_rows(points, N=n_sites)

    # get the correct ID for each electrode (like in JRCLUS)
    _ids = np.arange(n_sites + 1)
    ids = np.hstack([_ids[2::2], _ids[1::2]])

    # get the brain region of every electrode
    regions_metadata = pd.read_csv(
        Path(probe_metadata["reconstructed_track_filepath"]).with_suffix(
            ".csv"
        )
    ).iloc[::-1]
    regions_metadata = select_rows(regions_metadata, N=n_sites)

    # put everything together
    recording_sites = []
    for n in range(n_sites):
        if n >= len(electrodes):
            # site outside of brain
            recording_sites.append(
                {
                    "site_id": ids[n],
                    "registered_brain_coordinates": np.full(
                        3, fill_value=np.nan
                    ),
                    "brain_region": "OUT",
                    "brain_region_id": -1,
                    "probe_coordinates": int(n * 20),
                    "color": "k",
                }
            )
        else:
            try:
                rid = int(regions_metadata["Region ID"].iloc[n])
                acro = regions_metadata["Region acronym"].iloc[n]
                color_rgb = [
                    c / 255
                    for c in atlas._get_from_structure(acro, "rgb_triplet")
                ]
                color = rgb2hex(color_rgb)
            except ValueError:
                # not found in brain
                rid = -1
                acro = "OUT"
                color = "k"

            recording_sites.append(
                {
                    "site_id": ids[n],
                    "registered_brain_coordinates": electrodes[n],
                    "brain_region": acro,
                    "brain_region_id": rid,
                    "probe_coordinates": int(n * 20),
                    "color": color,
                }
            )

    return recording_sites
