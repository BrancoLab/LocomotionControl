import numpy as np
from pathlib import Path
from typing import List
from loguru import logger
from dataclasses import dataclass

from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from myterial.utils import rgb2hex


@dataclass
class ActiveElectrode:
    idx: int
    probe_position: int  # distance in um along the Y axis of the probe


def prepare_electrodes_positions(
    configuration: str, n_sites: int = 384
) -> List[ActiveElectrode]:
    """
        Defines the position along the probe (in coordinates from the first electrode)
        of each active electrode
    """
    if configuration == "b0":
        Y = 20 * np.repeat(np.arange(0, int(n_sites / 2)), 2)
        ids = np.arange(1, n_sites + 1)
    elif configuration == "longcolumn":
        Y = 20 * np.arange(n_sites)

        # odd numbers for bank 0 and even for bank 1
        _ids = np.arange(n_sites + 1)
        ids = np.hstack([_ids[1::2], _ids[2::2]])
    else:
        raise NotImplementedError(
            f'Probe configuration "{configuration}" not supported'
        )

    return [ActiveElectrode(idx, y) for idx, y in zip(ids, Y)]


def reconstructed_track_quality_check(
    implanted_depth: int, point_distances: np.ndarray
):
    """
        Checks that a reconstructed track is accurate by:
            1 - ensuring that the length of the reconstructed track matches the estimated insterted depth

    """
    # check that the length of the reconstructed track is comparable to that of the implanted depth
    if abs(point_distances[-1] - implanted_depth) > 100:
        logger.warning(
            f"""
            The probe was implanted at a depth of {implanted_depth} (in brain space) but the reconstructed track
            length is: {point_distances[-1]} (in atlas space).
        """
        )
        # return False
    return True


def place_probe_recording_sites(
    probe_metadata: dict,
    configuration: str,
    n_sites: int = 384,
    tip: int = 375,
) -> list:
    # get brainglobe atlas
    atlas = BrainGlobeAtlas("allen_mouse_25um")

    # find multiple reconstructions files paths
    try:
        mouse = probe_metadata["mouse_id"][-3:]
        files = Path(
            r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\reconstructed_probe_location"
        ).glob(mouse + "_atlas_space_0.npy")
        rec_path = list(files)[0]
    except IndexError:
        logger.warning(
            f"Did not find reconstructed track filepath for mouse {mouse}"
        )
        return

    # _, name = rec_path.parent, rec_path.stem
    # reconstruction_files = files(fld, pattern=f"{name}_*.npy")
    reconstruction_files = [rec_path]

    if reconstruction_files is None:
        logger.warning("Did not find any reconstruction files!")
        return
    elif not isinstance(reconstruction_files, list):
        reconstruction_files = [reconstruction_files]

    logger.debug(
        f"Identified {len(reconstruction_files)} reconstruction files"
    )

    # get the correct ID for each electrode (like in JRCLUS) & probe coords
    active_electrodes = prepare_electrodes_positions(configuration, n_sites)
    indices = [x.idx for x in active_electrodes]
    electrodes_coordinates = {id: [] for id in indices}

    # reconstruct electrodes positions for each file
    outside_brain = {idx: False for idx in indices}
    for rec_file in reconstruction_files:
        # load points (1 every um of inserted probe) and check that the right number of points is loaded
        try:
            points = np.load(rec_file)[
                ::-1
            ]  # flipped so that the first point is at the bottom of the probe like in brain
            # and exclude tip of probe
        except TypeError:
            logger.warning("Could not load reconstructed track file")
            return

        # get the distance of each point along the probe
        # the point corresponding to the first point on the track
        point_distances = np.apply_along_axis(
            np.linalg.norm, 1, points - points[0]
        ).astype(np.int32)

        # do quality check
        if not reconstructed_track_quality_check(
            probe_metadata["implanted_depth"], point_distances
        ):
            continue

        # exclude points on the tuip of the probe
        first_non_tip = np.where(point_distances > tip)[0][0]
        points = points[first_non_tip:]

        # compute points distances again but with 0 at first electrode now
        point_distances = np.apply_along_axis(
            np.linalg.norm, 1, points - points[0]
        ).astype(np.int32)

        # get the coordinates of each electrode
        for electrode in active_electrodes:
            # check if electrode is outside the brain
            if electrode.probe_position > np.max(point_distances):
                electrodes_coordinates[electrode.idx].append(
                    np.full(3, np.nan)
                )
                outside_brain[electrode.idx] = True

            # get the point closest to the electrode
            point_idx = np.argmin(
                abs(point_distances - electrode.probe_position)
            )
            electrodes_coordinates[electrode.idx].append(points[point_idx])

            if point_distances[point_idx] < -1:
                raise ValueError("Cant select points on the tip")

    # get the average coordinates for each electrode
    try:
        avg_electrode_coord = {
            k: np.nanmean(np.vstack(c), 0)
            for k, c in electrodes_coordinates.items()
        }
    except ValueError:
        logger.warning("Failed to reconstruct probe geometry")
        return
    sites_probe_coords = {e.idx: e.probe_position for e in active_electrodes}

    # get the brain region of each electrode
    recording_sites = []
    for e_idx, coords in avg_electrode_coord.items():
        # reconstruct position in brain
        if outside_brain[e_idx]:
            acro = "OUT"
            color = rgb2hex([0.1, 0.1, 0.1])
        else:
            rid = atlas.structure_from_coords(coords, microns=True)
            if rid == 0:
                acro = "unknown"
                color = rgb2hex([0.3, 0.3, 0.3])
            else:
                acro = atlas.structure_from_coords(
                    coords, microns=True, as_acronym=True
                )

                color = rgb2hex(
                    [
                        c / 255
                        for c in atlas._get_from_structure(acro, "rgb_triplet")
                    ]
                )

        recording_sites.append(
            {
                "site_id": e_idx,
                "registered_brain_coordinates": coords,
                "brain_region": acro,
                "brain_region_id": rid,
                "probe_coordinates": sites_probe_coords[e_idx],
                "color": color,
            }
        )

    # assign to undefined structures the closest well defined structure
    excluded = (
        "P",
        "MB",
        "scp",
        "II",
        "APr",
        "unknown",
        "PPY",
        "MRN",
        "P5",
        "MY",
        "scwm",
        "bic",
        "OUT",
        "tb",
    )
    for n, esite in enumerate(recording_sites):
        if esite["brain_region"] in excluded:
            shift = 1
            while True:
                if (
                    n + shift < len(recording_sites)
                    and recording_sites[n + shift]["brain_region"]
                    not in excluded
                ):
                    for key in ("brain_region", "brain_region_id", "color"):
                        esite[key] = recording_sites[n + shift][key]
                    break

                if (
                    n - shift > 0
                    and recording_sites[n - shift]["brain_region"]
                    not in excluded
                ):
                    for key in ("brain_region", "brain_region_id", "color"):
                        esite[key] = recording_sites[n - shift][key]
                    break

                if n + shift > len(recording_sites) and n - shift < 0:
                    break
                shift += 1

        elif (
            "RSP" not in esite["brain_region"]
            and "VISp" not in esite["brain_region"]
            and "ICe" not in esite["brain_region"]
        ):
            if n + 6 < len(recording_sites) and n > 6:
                for i in range(3):
                    i *= 2
                    if recording_sites[n + i]["brain_region"] in (
                        "PPN",
                        "CUN",
                        "GRN",
                    ):
                        if (
                            abs(
                                esite["probe_coordinates"]
                                - recording_sites[n + i]["probe_coordinates"]
                            )
                            < 75
                        ):
                            for key in (
                                "brain_region",
                                "brain_region_id",
                                "color",
                            ):
                                esite[key] = recording_sites[n + i][key]

                    elif recording_sites[n - i]["brain_region"] in (
                        "PPN",
                        "CUN",
                        "GRN",
                    ):
                        if (
                            abs(
                                esite["probe_coordinates"]
                                - recording_sites[n - i]["probe_coordinates"]
                            )
                            < 75
                        ):
                            for key in (
                                "brain_region",
                                "brain_region_id",
                                "color",
                            ):
                                esite[key] = recording_sites[n - i][key]

        recording_sites[n] = esite

    if len(recording_sites) != n_sites:
        raise ValueError(
            f"Expected {n_sites} recordiing sites, found: {len(recording_sites)}"
        )

    return recording_sites
