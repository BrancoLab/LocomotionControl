import numpy as np
import pandas as pd
from scipy.stats import circmean

from fcutils.file_io.utils import check_file_exists
from fcutils.maths.filtering import median_filter_1d
from fcutils.maths.geometry import (
    calc_angle_between_vectors_of_points_2d as get_bone_angle,
)
from fcutils.maths.geometry import (
    calc_distance_between_points_in_a_vector_2d as get_speed_from_xy,
)
from fcutils.maths.geometry import (
    calc_angle_between_points_of_vector_2d as get_dir_of_mvmt_from_xy,
)
from fcutils.maths.geometry import calc_ang_velocity
from fcutils.maths.geometry import calc_distance_between_points_2d

# ---------------------------------------------------------------------------- #
#                                    ANGLES                                    #
# ---------------------------------------------------------------------------- #


def average_body_angle(*angles, deg=True):
    """
        Given a variable number of 1d numpy array of the same size, 
        take the angular average at each frame.
    """
    if deg:
        angles = [np.radians(a) for a in angles]
    angs = np.vstack(angles)
    mean_angle = [circmean(angs[:, i]) for i in range(angs.shape[1])]
    return mean_angle if not deg else np.degrees(mean_angle)


def ang_difference(a1, a2, deg=True):
    """Compute the smallest difference between two angle arrays.
    Parameters
    ----------
    a1, a2 : np.ndarray
        The angle arrays to subtract
    deg : bool (default=False)
        Whether to compute the difference in degrees or radians
    Returns
    -------
    out : np.ndarray
        The difference between a1 and a2
    """

    diff = a1 - a2
    return wrapdiff(diff, deg=deg)


def wrapdiff(diff, deg=True):
    """Given an array of angle differences, make sure that they lie
    between -pi and pi.
    Parameters
    ----------
    diff : np.ndarray
        The angle difference array
    deg : bool (default=False)
        Whether the angles are in degrees or radians
    Returns
    -------
    out : np.ndarray
        The updated angle differences
    """

    if deg:
        base = 360
    else:
        base = np.pi * 2

    i = np.abs(diff) > (base / 2.0)
    out = diff.copy()
    out[i] -= np.sign(diff[i]) * base
    return out


# ---------------------------------------------------------------------------- #
#                                      DLC                                     #
# ---------------------------------------------------------------------------- #


def get_scorer_bodyparts(tracking):
    """
        Given the tracking data hierarchical df from DLC, return
        the scorer and bodyparts names
    """
    first_frame = tracking.iloc[0]
    try:
        bodyparts = first_frame.index.levels[1]
        scorer = first_frame.index.levels[0]
    except:
        raise NotImplementedError(
            "Make this return something helpful when not DLC df"
        )

    return scorer, bodyparts


def clean_dlc_tracking(tracking):
    """
        Given the tracking data hierarchical df from DLC, 
        returns a simplified version of it. 
    """
    scorer, bodyparts = get_scorer_bodyparts(tracking)
    tracking = tracking.unstack()

    trackings = {}
    for bp in bodyparts:
        tr = {
            c: tracking.loc[scorer, bp, c].values
            for c in ["x", "y", "likelihood"]
        }
        trackings[bp] = pd.DataFrame(tr)

    return trackings, bodyparts


def prepare_tracking_data(
    tracking_filepath=None,
    tracking=None,
    bodyparts=None,
    likelihood_th=0.999,
    median_filter=False,
    filter_kwargs={},
    compute=True,
    smooth_dir_mvmt=True,
    interpolate_nans=True,
    verbose=False,
):
    """
        Loads, cleans and filters tracking data from dlc.
        Also handles fisheye correction and registration to common coordinates frame.
        Can be used to compute speeds and angles for each bp.

        :params tracking: pd.DataFrame. Optional, pass a dataframe with tracking data
            else pass  a file to tracking filepath to load from DLC output
        :param tracking_filepath: path to file to process
        :param likelihood_th: float, frames with likelihood < thresh are nanned
        :param median_filter: if true the data are filtered before the processing
        :param filter_kwargs: arguments for median filtering func
        :param compute: if true speeds and angles are computed
        :param smooth_dir_mvmt: if true the direction of mvmt is smoothed with a median filt.
        :param interpolate_nans: if true it removes nans from the tracking data by linear interpolation
    """

    # Load the tracking data
    if tracking_filepath is not None:
        check_file_exists(tracking_filepath, raise_error=True)
        if ".h5" not in tracking_filepath:
            raise ValueError("Expected .h5 in the tracking data file path")

        if verbose:
            print("Processing: {}".format(tracking_filepath))
        tracking, bodyparts = clean_dlc_tracking(
            pd.read_hdf(tracking_filepath)
        )
    elif tracking is None or bodyparts is None:
        raise ValueError(
            "Pass either tracking_filepath or tracking+body parts"
        )

    # Get likelihood and XY coords
    likelihoods = {}
    for bp in bodyparts:
        likelihoods[bp] = tracking[bp]["likelihood"].values
        tracking[bp].drop("likelihood", axis=1)

    # Median filtering
    if median_filter:
        if verbose:
            print("     applying median filter")
        for bp in bodyparts:
            tracking[bp]["x"] = median_filter_1d(
                tracking[bp]["x"].values, **filter_kwargs
            )
            tracking[bp]["y"] = median_filter_1d(
                tracking[bp]["y"].values, **filter_kwargs
            )

    # Compute speed, angular velocity etc...
    if compute:
        if verbose:
            print("     computing speeds and angles")
        for bp in bodyparts:
            x, y = tracking[bp].x.values, tracking[bp].y.values

            tracking[bp]["speed"] = get_speed_from_xy(x, y)

            if not smooth_dir_mvmt:
                tracking[bp][
                    "direction_of_movement"
                ] = get_dir_of_mvmt_from_xy(x, y)
            else:
                tracking[bp]["direction_of_movement"] = median_filter_1d(
                    get_dir_of_mvmt_from_xy(x, y), kernel=41
                )

            tracking[bp]["angular_velocity"] = calc_ang_velocity(
                tracking[bp]["direction_of_movement"].values
            )

    # Remove nans
    for bp, like in likelihoods.items():
        tracking[bp][like < likelihood_th] = np.nan

        if interpolate_nans:
            tracking[bp] = tracking[bp].interpolate(axis=0)
    return tracking


def compute_body_segments(tracking, segments):
    """ 
        Given a dictionary of dataframes with tracking and a list of bones (body segments) it computes stuff on the bones
        and returns the results

        :param tracking: dictionary of dataframes with tracking for each bodypart
        :param segments: dict of two-tuples. Keys are the names of the bones and tuple elements the 
                names of the bodyparts that define each bone.

    """
    bones = {}
    for bone, (bp1, bp2) in segments.items():
        # get the XY tracking data
        bp1, bp2 = tracking[bp1], tracking[bp2]

        # Get bone orientation
        bone_orientation = get_bone_angle(
            bp1.x.values, bp1.y.values, bp2.x.values, bp2.y.values,
        )

        # Get bone length [first remove nans to allow computation]
        bp1_tr = np.array([bp1.x.values, bp1.y.values]).T
        bp2_tr = np.array([bp2.x.values, bp2.y.values]).T

        nan_idxs = (
            list(np.where(np.isnan(bp1_tr[:, 0]))[0])
            + list(np.where(np.isnan(bp1_tr[:, 1]))[0])
            + list(np.where(np.isnan(bp2_tr[:, 0]))[0])
            + list(np.where(np.isnan(bp2_tr[:, 1]))[0])
        )

        bone_length = np.array(
            [
                calc_distance_between_points_2d(p1, p2)
                for p1, p2 in zip(np.nan_to_num(bp1_tr), np.nan_to_num(bp2_tr))
            ]
        )
        bone_length[nan_idxs] = np.nan  # replace nans

        # Put everything together
        bones[bone] = pd.DataFrame(
            dict(bone_length=bone_length, bone_orientation=bone_orientation,)
        )
    return bones
