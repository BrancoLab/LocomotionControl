import sys

sys.path.append("./")

import numpy as np

from data.dbase.db_tables import Tracking

# Kallmann filtering of XY tracking data from: https://github.com/joacorapela/lds_python


def smoothLDS_SS(B, xnn, Vnn, xnn1, Vnn1, m0, V0):
    """ Kalman smoother implementation
    :param: B: state transition matrix
    :type: B: numpy matrix (MxM)
    :param: xnn: filtered means (from Kalman filter)
    :type: xnn: numpy array (MxT)
    :param: Vnn: filtered covariances (from Kalman filter)
    :type: Vnn: numpy array (MxMXT)
    :param: xnn1: predicted means (from Kalman filter)
    :type: xnn1: numpy array (MxT)
    :param: Vnn1: predicted covariances (from Kalman filter)
    :type: Vnn1: numpy array (MxMXT)
    :param: m0: initial state mean
    :type: m0: one-dimensional numpy array (M)
    :param: V0: initial state covariance
    :type: V0: numpy matrix (MxM)
    :return:  {xnN, VnN, Jn, x0N, V0N, J0}: xnn1 and Vnn1 (smoothed means, MxT, and covariances, MxMxT), Jn (smoothing gain matrix, MxMxT), x0N and V0N (smoothed initial state mean, M, and covariance, MxM), J0 (initial smoothing gain matrix, MxN).
    """
    N = xnn.shape[2]
    M = B.shape[0]
    xnN = np.empty(shape=[M, 1, N])
    VnN = np.empty(shape=[M, M, N])
    Jn = np.empty(shape=[M, M, N])

    xnN[:, :, -1] = xnn[:, :, -1]
    VnN[:, :, -1] = Vnn[:, :, -1]
    for n in reversed(range(1, N)):
        Jn[:, :, n - 1] = Vnn[:, :, n - 1] @ B.T @ np.linalg.inv(Vnn1[:, :, n])
        xnN[:, :, n - 1] = xnn[:, :, n - 1] + Jn[:, :, n - 1] @ (
            xnN[:, :, n] - xnn1[:, :, n]
        )
        VnN[:, :, n - 1] = (
            Vnn[:, :, n - 1]
            + Jn[:, :, n - 1]
            @ (VnN[:, :, n] - Vnn1[:, :, n])
            @ Jn[:, :, n - 1].T
        )
    # initial state x00 and V00
    # return the smooth estimates of the state at time 0: x0N and V0N
    J0 = V0 @ B.T @ np.linalg.inv(Vnn1[:, :, 0])
    x0N = m0 + J0 @ (xnN[:, :, 0] - xnn1[:, :, 0])
    V0N = V0 + J0 @ (VnN[:, :, 0] - Vnn1[:, :, 0]) @ J0.T
    answer = {
        "xnN": xnN,
        "VnN": VnN,
        "Jn": Jn,
        "x0N": x0N,
        "V0N": V0N,
        "J0": J0,
    }
    return answer


def filterLDS_SS_withMissingValues_np(y, B, Q, m0, V0, Z, R):
    """ Kalman filter implementation of the algorithm described in Shumway and
    Stoffer 2006.
    :param: y: time series to be smoothed
    :type: y: numpy array (NxT)
    :param: B: state transition matrix
    :type: B: numpy matrix (MxM)
    :param: Q: state noise covariance matrix
    :type: Q: numpy matrix (MxM)
    :param: m0: initial state mean
    :type: m0: one-dimensional numpy array (M)
    :param: V0: initial state covariance
    :type: V0: numpy matrix (MxM)
    :param: Z: state to observation matrix
    :type: Z: numpy matrix (NxM)
    :param: R: observations covariance matrix
    :type: R: numpy matrix (NxN)
    :return:  {xnn1, Vnn1, xnn, Vnn, innov, K, Sn, logLike}: xnn1 and Vnn1 (predicted means, MxT, and covariances, MxMxT), xnn and Vnn (filtered means, MxT, and covariances, MxMxT), innov (innovations, NxT), K (Kalman gain matrices, MxNxT), Sn (innovations covariance matrices, NxNxT), logLike (data loglikelihood, float).
    :rtype: dictionary
    """

    # N: number of observations
    # M: dim state space
    # P: dim observations
    M = B.shape[0]
    N = y.shape[1]
    P = y.shape[0]
    xnn1 = np.empty(shape=[M, 1, N])
    Vnn1 = np.empty(shape=[M, M, N])
    xnn = np.empty(shape=[M, 1, N])
    Vnn = np.empty(shape=[M, M, N])
    innov = np.empty(shape=[P, 1, N])
    Sn = np.empty(shape=[P, P, N])

    # k==0
    xnn1[:, 0, 0] = (B @ m0).squeeze()
    Vnn1[:, :, 0] = B @ V0 @ B.T + Q
    Stmp = Z @ Vnn1[:, :, 0] @ Z.T + R
    Sn[:, :, 0] = (Stmp + Stmp.T) / 2
    Sinv = np.linalg.inv(Sn[:, :, 0])
    K = Vnn1[:, :, 0] @ Z.T @ Sinv
    innov[:, 0, 0] = y[:, 0] - (Z @ xnn1[:, :, 0]).squeeze()
    xnn[:, :, 0] = xnn1[:, :, 0] + K @ innov[:, :, 0]
    Vnn[:, :, 0] = Vnn1[:, :, 0] - K @ Z @ Vnn1[:, :, 0]
    logLike = (
        -N * P * np.log(2 * np.pi)
        - np.linalg.slogdet(Sn[:, :, 0])[1]
        - innov[:, :, 0].T @ Sinv @ innov[:, :, 0]
    )

    # k>1
    for k in range(1, N):
        xnn1[:, :, k] = B @ xnn[:, :, k - 1]
        Vnn1[:, :, k] = B @ Vnn[:, :, k - 1] @ B.T + Q
        if np.any(np.isnan(y[:, k])):
            xnn[:, :, k] = xnn1[:, :, k]
            Vnn[:, :, k] = Vnn1[:, :, k]
        else:
            Stmp = Z @ Vnn1[:, :, k] @ Z.T + R
            Sn[:, :, k] = (Stmp + Stmp.T) / 2
            Sinv = np.linalg.inv(Sn[:, :, k])
            K = Vnn1[:, :, k] @ Z.T @ Sinv
            innov[:, 0, k] = y[:, k] - (Z @ xnn1[:, :, k]).squeeze()
            xnn[:, :, k] = xnn1[:, :, k] + K @ innov[:, :, k]
            Vnn[:, :, k] = Vnn1[:, :, k] - K @ Z @ Vnn1[:, :, k]
        logLike = (
            logLike
            - np.linalg.slogdet(Sn[:, :, k])[1]
            - innov[:, :, k].T @ Sinv @ innov[:, :, k]
        )
    logLike = 0.5 * logLike
    answer = {
        "xnn1": xnn1,
        "Vnn1": Vnn1,
        "xnn": xnn,
        "Vnn": Vnn,
        "innov": innov,
        "KN": K,
        "Sn": Sn,
        "logLike": logLike,
    }
    return answer


# ---------------------------------------------------------------------------- #
#                                   LOAD DATA                                  #
# ---------------------------------------------------------------------------- #

data = Tracking.get_session_tracking(
    "FC_210722_AAA1110750_hairpin", body_only=False
)

T = 1000
x, y = data.x.iloc[0], data.y.iloc[0]
xy = np.vstack([x, y])[:, :T]


def kalmann(
    xy: np.ndarray,
    sigma_a=0.1,  # standard deviation of acceleration
    sigma_x=0.001,  # standard deviation of x coordinate of position
    sigma_y=0.001,  # standard deviation of y coordinate of position
    V0_diag_value0=0.001,  # initial state variance
    dt=1 / 60,
) -> dict:

    """
        Does Kalmann filtering/smoothing as described in: https://github.com/joacorapela/lds_python

        Arguments:
            xy: numpy array, 2xT with XY coordinates of a body part at each frame
    
        Returns:
            a dictionary with x,y coordiantes, speed and accelerations.
            Since it uses `dt` the speed and acceleration quantities are alrady in cm/s
    """

    # ------------------------------ BUILD MATRICES ------------------------------ #

    # Taken from the book
    # barShalomEtAl01-estimationWithApplicationToTrackingAndNavigation.pdf
    # section 6.3.3
    # Eq. 6.3.3-2
    B = np.array(
        [
            [1, dt, 0.5 * dt ** 2, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, 0.5 * dt ** 2],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.double,
    )
    Z = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=np.double)
    # Eq. 6.3.3-4
    Q = (
        np.array(
            [
                [dt ** 4 / 4, dt ** 3 / 2, dt ** 2 / 2, 0, 0, 0],
                [dt ** 3 / 2, dt ** 2, dt, 0, 0, 0],
                [dt ** 2 / 2, dt, 1, 0, 0, 0],
                [0, 0, 0, dt ** 4 / 4, dt ** 3 / 2, dt ** 2 / 2],
                [0, 0, 0, dt ** 3 / 2, dt ** 2, dt],
                [0, 0, 0, dt ** 2 / 2, dt, 1],
            ],
            dtype=np.double,
        )
        * sigma_a ** 2
    )
    R = np.diag([sigma_x ** 2, sigma_y ** 2])

    m0 = np.array([xy[0, 0], 0, 0, xy[1, 0], 0, 0], dtype=np.double)
    V0 = np.diag(np.ones(len(m0)) * V0_diag_value0)

    # ----------------------------- FILTER AND SMOOTH ---------------------------- #
    filterRes = filterLDS_SS_withMissingValues_np(
        y=xy, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R
    )
    smoothRes = smoothLDS_SS(
        B=B,
        xnn=filterRes["xnn"],
        Vnn=filterRes["Vnn"],
        xnn1=filterRes["xnn1"],
        Vnn1=filterRes["Vnn1"],
        m0=m0,
        V0=V0,
    )

    # prep results
    res = dict(
        x=smoothRes["xnN"][0, 0, :],
        xdot=smoothRes["xnN"][1, 0, :],
        xdotdot=smoothRes["xnN"][2, 0, :],
        y=smoothRes["xnN"][3, 0, :],
        ydot=smoothRes["xnN"][4, 0, :],
        ydotdot=smoothRes["xnN"][5, 0, :],
    )
    return res
