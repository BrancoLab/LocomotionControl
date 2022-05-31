import cv2
import numpy as np
from loguru import logger
from pathlib import Path


import fcutils.video as video_utils

from data import paths

TEMPLATE_POINTS = np.array(
    [[40, 58], [645, 58], [645, 930], [40, 930]]
)  # TO DEFINE


def get_matrix(videopath, template):
    """
        Loads or creates a registration matrix
    """
    # try to load the matrix first
    save_path = (paths.ccm_matrices / Path(videopath).name).with_suffix(".npy")
    if save_path.exists():
        logger.debug(f"Found CCM matrix at: {save_path}")
        return np.load(save_path)

    # didn't find a matrix, make one manually
    # Get the background (first frame) of the video being processed
    try:
        cap = cv2.VideoCapture(videopath)
        if not cap.isOpened():
            ValueError
    except:
        raise FileNotFoundError("Couldnt open file ", videopath)

    cap = video_utils.get_cap_from_file(videopath)
    if cap is None:
        raise FileNotFoundError(f"Failed to open video at: {videopath}")

    frame = video_utils.get_cap_selected_frame(cap, 0)

    # manually create registration matrix
    M = create_matrix(frame, template, TEMPLATE_POINTS, save_path)
    try:
        np.save(save_path, M)
    except FileNotFoundError:
        logger.warning(f"Could not save CCM matrix at: {save_path}")
    return M


# =================================================================================
#              IMAGE REGISTRATION GUI
# =================================================================================
def create_matrix(background, arena, arena_points, save_path):
    """
        Manual GUI for creating a registratoin matrix. 
        Credit to Philip Shamash (Branco Lab) -  https://github.com/BrancoLab/Common-Coordinate-Behaviour
    """
    background = cv2.copyMakeBorder(
        background, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    if np.any(np.array(background.shape) > 1000):
        background = cv2.resize(
            background,
            (int(background.shape[1] / 2), int(background.shape[0] / 2)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        background = cv2.resize(
            background,
            (background.shape[1], background.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)

    # initialize clicked points
    blank_arena = arena.copy()
    blank_arena = cv2.resize(
        blank_arena, background.T.shape, interpolation=cv2.INTER_AREA
    )
    print(
        "\nBackground and blank arena shape:",
        background.shape,
        blank_arena.shape,
    )

    background_data = [background, np.array(([], [])).T]
    arena_data = [[], np.array(([], [])).T]
    cv2.namedWindow("registered background")
    alpha = 0.7
    colors = [[150, 0, 150], [0, 255, 0]]
    color_array = make_color_array(colors, background.shape)

    # let user click points
    print(
        "\nSelect reference points on the experimental background image in the indicated order"
    )
    # initialize clicked point arrays
    background_data = [background, np.array(([], [])).T]
    arena_data = [[], np.array(([], [])).T]

    # add 1-2-3-4 markers to model arena
    for i, point in enumerate(arena_points.astype(np.uint32)):
        arena = cv2.circle(arena, (point[0], point[1]), 3, [0, 0, 255], -1)
        arena = cv2.circle(arena, (point[0], point[1]), 4, [0, 255, 255], 1)
        cv2.putText(
            arena,
            str(i + 1),
            tuple(point),
            0,
            0.55,
            [0, 255, 255],
            thickness=2,
        )

        point = np.reshape(point, (1, 2))
        arena_data[1] = np.concatenate((arena_data[1], point))

    # initialize GUI
    cv2.startWindowThread()
    cv2.namedWindow("background")
    cv2.imshow("background", background)
    cv2.namedWindow("model arena")
    cv2.imshow("model arena", arena)

    # create functions to react to clicked points
    cv2.setMouseCallback(
        "background", select_transform_points, background_data
    )  # Mouse callback

    while True:  # take in clicked points until four points are clicked
        cv2.imshow("background", background)

        number_clicked_points = background_data[1].shape[0]
        if number_clicked_points == len(arena_data[1]):
            break
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # perform projective transform & register background
    M = cv2.estimateAffine2D(background_data[1], arena_data[1], False)[0]
    if not M.any():
        raise ValueError("Could not calculate Rigid Transform")
    registered_background = cv2.warpAffine(
        background, M, (background.shape[1], background.shape[0])
    )
    print(f"Registered background shape: {registered_background.shape}")

    # --------------------------------------------------
    # overlay images
    # --------------------------------------------------
    alpha = 0.7
    colors = [[150, 0, 150], [0, 255, 0]]
    color_array = make_color_array(colors, blank_arena.shape)

    registered_background_color = (
        cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
        * np.squeeze(color_array[:, :, :, 0])
    ).astype(np.uint8)
    arena_color = (
        cv2.cvtColor(blank_arena, cv2.COLOR_GRAY2RGB)
        * np.squeeze(color_array[:, :, :, 1])
    ).astype(np.uint8)

    overlaid_arenas = cv2.addWeighted(
        registered_background_color, alpha, arena_color, 1 - alpha, 0
    )
    cv2.imshow("registered background", overlaid_arenas)

    # --------------------------------------------------
    # initialize GUI for correcting transform
    # --------------------------------------------------
    print(
        """
        \nLeft click model arena // Right click model background
        Purple within arena and green along the boundary represent the model arena
        Advanced users: use arrow keys and \'wasd\' to fine-tune translation and scale as a final step
        Crème de la crème: use \'tfgh\' to fine-tune shear\n
        y: save and use transform
        r: reset transform (left and right click four points to recommence)
    """
    )
    update_transform_data = [
        overlaid_arenas,
        background_data[1],
        arena_data[1],
        M,
        background,
    ]

    # create functions to react to additional clicked points
    cv2.setMouseCallback(
        "registered background",
        additional_transform_points,
        update_transform_data,
    )

    # take in clicked points until 'q' is pressed
    initial_number_clicked_points = [
        update_transform_data[1].shape[0],
        update_transform_data[2].shape[0],
    ]
    M_initial = M
    M_indices = [
        (0, 2),
        (1, 2),
        (0, 0),
        (1, 1),
        (0, 1),
        (1, 0),
        (2, 0),
        (2, 2),
    ]
    # M_indices_meanings = ['x-translate','y-translate','x-scale','y-scale','x->y shear','y->x shear','x perspective','y perspective']
    M_mod_keys = [
        2424832,
        2555904,
        2490368,
        2621440,
        ord("w"),
        ord("a"),
        ord("s"),
        ord("d"),
        ord("f"),
        ord("t"),
        ord("g"),
        ord("h"),
        ord("j"),
        ord("i"),
        ord("k"),
        ord("l"),
    ]
    while True:
        cv2.imshow("registered background", overlaid_arenas)
        cv2.imshow("background", registered_background)
        number_clicked_points = [
            update_transform_data[1].shape[0],
            update_transform_data[2].shape[0],
        ]
        update_transform = False
        k = cv2.waitKey(10)
        # If a left and right point are clicked:
        if (
            number_clicked_points[0] > initial_number_clicked_points[0]
            and number_clicked_points[1] > initial_number_clicked_points[1]
        ):
            initial_number_clicked_points = number_clicked_points
            # update transform and overlay images
            try:
                # M = cv2.findHomography(update_transform_data[1], update_transform_data[2])
                M = cv2.estimateRigidTransform(
                    update_transform_data[1], update_transform_data[2], False
                )  # True ~ full transform
                update_transform = True
            except:
                continue
        elif k in M_mod_keys:  # if an arrow key if pressed
            if k == 2424832:  # left arrow - x translate
                M[M_indices[0]] = (
                    M[M_indices[0]] - abs(M_initial[M_indices[0]]) * 0.005
                )
            elif k == 2555904:  # right arrow - x translate
                M[M_indices[0]] = (
                    M[M_indices[0]] + abs(M_initial[M_indices[0]]) * 0.005
                )
            elif k == 2490368:  # up arrow - y translate
                M[M_indices[1]] = (
                    M[M_indices[1]] - abs(M_initial[M_indices[1]]) * 0.005
                )
            elif k == 2621440:  # down arrow - y translate
                M[M_indices[1]] = (
                    M[M_indices[1]] + abs(M_initial[M_indices[1]]) * 0.005
                )
            elif k == ord("a"):  # down arrow - x scale
                M[M_indices[2]] = (
                    M[M_indices[2]] + abs(M_initial[M_indices[2]]) * 0.005
                )
            elif k == ord("d"):  # down arrow - x scale
                M[M_indices[2]] = (
                    M[M_indices[2]] - abs(M_initial[M_indices[2]]) * 0.005
                )
            elif k == ord("s"):  # down arrow - y scale
                M[M_indices[3]] = (
                    M[M_indices[3]] + abs(M_initial[M_indices[3]]) * 0.005
                )
            elif k == ord("w"):  # down arrow - y scale
                M[M_indices[3]] = (
                    M[M_indices[3]] - abs(M_initial[M_indices[3]]) * 0.005
                )
            elif k == ord("f"):  # down arrow - x-y shear
                M[M_indices[4]] = (
                    M[M_indices[4]] - abs(M_initial[M_indices[4]]) * 0.005
                )
            elif k == ord("h"):  # down arrow - x-y shear
                M[M_indices[4]] = (
                    M[M_indices[4]] + abs(M_initial[M_indices[4]]) * 0.005
                )
            elif k == ord("t"):  # down arrow - y-x shear
                M[M_indices[5]] = (
                    M[M_indices[5]] - abs(M_initial[M_indices[5]]) * 0.005
                )
            elif k == ord("g"):  # down arrow - y-x shear
                M[M_indices[5]] = (
                    M[M_indices[5]] + abs(M_initial[M_indices[5]]) * 0.005
                )

            update_transform = True

        elif k == ord("r"):
            print("Transformation erased")
            update_transform_data[1] = np.array(([], [])).T
            update_transform_data[2] = np.array(([], [])).T
            initial_number_clicked_points = [3, 3]
        elif k == ord("q") or k == ord("y"):
            print("Registration completed")
            break

        if update_transform:
            update_transform_data[3] = M
            # registered_background = cv2.warpPerspective(background, M, background.shape)
            registered_background = cv2.warpAffine(
                background, M, (background.shape[1], background.shape[0])
            )
            registered_background_color = (
                cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                * np.squeeze(color_array[:, :, :, 0])
            ).astype(np.uint8)
            overlaid_arenas = cv2.addWeighted(
                registered_background_color, alpha, arena_color, 1 - alpha, 0
            )
            update_transform_data[0] = overlaid_arenas

    cv2.destroyAllWindows()
    return M


# mouse callback function I
def select_transform_points(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 3, 255, -1)
        data[0] = cv2.circle(data[0], (x, y), 4, 0, 1)

        clicks = np.reshape(np.array([x, y]), (1, 2))
        data[1] = np.concatenate((data[1], clicks))


# mouse callback function II
def additional_transform_points(event, x, y, flags, data):
    if event == cv2.EVENT_RBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 2, (200, 0, 0), -1)
        data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

        M_inverse = cv2.invertAffineTransform(data[3])
        transformed_clicks = np.matmul(
            np.append(M_inverse, np.zeros((1, 3)), 0), [x, y, 1]
        )
        # M_inverse = np.linalg.inv(data[3])
        # M_inverse = cv2.findHomography(data[2][:len(data[1])], data[1])
        # transformed_clicks = np.matmul(M_inverse[0], [x, y, 1])

        data[4] = cv2.circle(
            data[4],
            (int(transformed_clicks[0]), int(transformed_clicks[1])),
            2,
            (0, 0, 200),
            -1,
        )
        data[4] = cv2.circle(
            data[4],
            (int(transformed_clicks[0]), int(transformed_clicks[1])),
            3,
            0,
            1,
        )

        clicks = np.reshape(transformed_clicks[0:2], (1, 2))
        data[1] = np.concatenate((data[1], clicks))
    elif event == cv2.EVENT_LBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 2, (0, 200, 200), -1)
        data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

        clicks = np.reshape(np.array([x, y]), (1, 2))
        data[2] = np.concatenate((data[2], clicks))


def make_color_array(colors, image_size):
    color_array = np.zeros(
        (image_size[0], image_size[1], 3, len(colors))
    )  # create coloring arrays
    for c in range(len(colors)):
        for i in range(3):  # B, G, R
            color_array[:, :, i, c] = (
                np.ones((image_size[0], image_size[1]))
                * colors[c][i]
                / sum(colors[c])
            )
    return color_array
