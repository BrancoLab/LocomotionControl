import numpy as np
from pathlib import Path
import pandas as pd
from typing import Union
from loguru import logger

from fcutils.path import files
from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from myterial.utils import rgb2hex


def select_rows(
    data: Union[pd.DataFrame, np.array],
    skip: int = 175,
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

    # find multiple reconstructions files paths
    try:
        rec_path = Path(probe_metadata["reconstructed_track_filepath"])
    except  TypeError:
        logger.warning(f'Did not find reconstructed track filepath')
        return

    fld, name = rec_path.parent, rec_path.stem
    reconstruction_files = files(fld, pattern=f"{name}_*.npy")

    if reconstruction_files is None:
        logger.warning("Did not find any reconstruction files!")
        return
    elif not isinstance(reconstruction_files, list):
        reconstruction_files = [reconstruction_files]

    logger.debug(
        f"Identified {len(reconstruction_files)} reconstruction files"
    )

    # get the correct ID for each electrode (like in JRCLUS)
    _ids = np.arange(n_sites + 1)
    ids = np.hstack([_ids[1::2], _ids[2::2]])
    electrodes_coordinates = {id: [] for id in ids}

    # reconstruct electrodes positions for each file
    for rec_file in reconstruction_files:
        # load points (1 every um of inserted probe)
        try:
            points = np.load(rec_file)
        except TypeError:
            logger.warning("Could not load reconstructed track file")
            return

        if abs(probe_metadata["implanted_depth"] * 1000 - len(points)) > 100:
            logger.warning(
                "The reconstructed probe should have one point for every um of inserted probe - too large delta"
            )
            return

        # discard first 200um because there's no electrodes threshold
        # one electrode every 20um
        # keep 384 electrodes
        electrodes = select_rows(points, N=n_sites)
        for id, electrode in zip(ids, electrodes):
            electrodes_coordinates[id].append(electrode)

    # Get the average position
    average_coordinates = {}
    for eid, epoints in electrodes_coordinates.items():
        if not len(epoints):
            continue
        average_coordinates[eid] = np.vstack(epoints).mean(axis=0)
    electrode_coordinates = np.vstack(average_coordinates.values())[::-1]

    # reconstruct brain region and color for each electrode
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
            coords = electrode_coordinates[n]
            rid = atlas.structure_from_coords(coords, microns=True)

            if rid == 0:
                acro = "unknown"
                color = rgb2hex([0.1, 0.1, 0.1])
            else:
                acro = atlas.structure_from_coords(
                    coords, microns=True, as_acronym=True
                )
                color_rgb = [
                    c / 255
                    for c in atlas._get_from_structure(acro, "rgb_triplet")
                ]
                color = rgb2hex(color_rgb)

            recording_sites.append(
                {
                    "site_id": ids[n],
                    "registered_brain_coordinates": coords,
                    "brain_region": acro,
                    "brain_region_id": rid,
                    "probe_coordinates": int(n * 20),
                    "color": color,
                }
            )

    return recording_sites
