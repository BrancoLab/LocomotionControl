import brainrender as br
import numpy as np
import pandas as pd
from pathlib import Path

from myterial import blue_grey

br.settings.SHOW_AXES = False


def render_probe_3d(
    rsites: pd.DataFrame, save_path: Path = None, targets: tuple = None
):
    targets = list(targets) or []

    # create brainrender scene
    scene = br.Scene(screenshots_folder=save_path)

    # add probe track
    track = np.vstack(rsites.registered_brain_coordinates.values)
    colors = [
        color
        if region in targets
        else (blue_grey if region not in ("unknown", "OUT") else "k")
        for color, region in zip(
            rsites.color.values, rsites.brain_region.values
        )
    ]
    pts = scene.add(br.actors.Points(track, colors=colors, radius=15))
    scene.add_silhouette(pts, lw=2)

    # add brain regions
    for region in rsites.brain_region.unique():
        if region in targets:
            alpha = 0.4
            targets.pop(targets.index(region))
        else:
            alpha = 0.01
        scene.add_brain_region(region, alpha=alpha)

    # add remainning targets
    if targets:
        scene.add_brain_region(*targets, alpha=0.4)

    # slice
    plane = scene.atlas.get_plane(
        norm=(0, 0, 1), pos=scene.root._mesh.centerOfMass()
    )
    scene.slice(plane)

    # render
    cam = {
        "pos": (7196, 247, -38602),
        "viewup": (0, -1, 0),
        "clippingRange": (29133, 44003),
        "focalPoint": (7718, 4290, -3507),
        "distance": 35331,
    }

    if save_path is None:
        scene.render(camera=cam)
    else:
        scene.render(camera=cam, interactive=False)
        scene.screenshot("activity_probe_rendering")
    scene.close()
    del scene


def render_probe_regions_slices(
    rsites: pd.DataFrame, save_path: Path = None, targets: tuple = None
):
    targets = list(targets) if targets is not None else []

    for target_region in rsites.brain_region.unique():
        for side in ("frontal", "sagittal2", "top"):
            scene_targets = targets.copy()
            scene = br.Scene(title=target_region, screenshots_folder=save_path)

            # add probe track
            track = np.vstack(rsites.registered_brain_coordinates.values)
            track_actor = scene.add(
                br.actors.Points(track, colors="k", radius=80)
            )

            # add brain regions
            actors = [track_actor]
            for region in rsites.brain_region.unique():
                if region == target_region:
                    alpha = 0.9
                elif region in scene_targets:
                    scene_targets.pop(scene_targets.index(region))
                    alpha = 0.5
                else:
                    alpha = 0.05

                act = scene.add_brain_region(region, alpha=alpha)
                if act is not None:
                    actors.append(act)

            # add remaining target regions
            if scene_targets:
                actors.extend(
                    scene.add_brain_region(*scene_targets, alpha=0.4)
                )

            # slice
            coords = np.mean(
                np.vstack(
                    rsites.loc[
                        rsites.brain_region == target_region
                    ].registered_brain_coordinates.values
                ),
                axis=0,
            )
            if side == "frontal":
                shift = np.array([250, 0, 0])
            elif side == "top":
                shift = np.array([0, 250, 0])
            else:
                coords[2] = -coords[2]
                shift = np.array([0, 0, 500])
            p1 = coords - shift
            p2 = coords + shift

            for p, norm in zip((p1, p2), (1, -1)):
                if side == "frontal":
                    nrm = (norm, 0, 0)
                elif side == "top":
                    nrm = (0, norm, 0)
                else:
                    nrm = (0, 0, norm)
                plane = scene.atlas.get_plane(pos=p, norm=nrm)
                scene.slice(plane, actors=actors, close_actors=True)
                scene.slice(plane, actors=scene.root, close_actors=False)

            # show/save
            if save_path is None:
                scene.render(camera=side)
            else:
                scene.render(camera=side, interactive=False)
                scene.screenshot(
                    f"activity_probe_slice_{target_region}_{side}"
                )
            scene.close()
            del scene


if __name__ == "__main__":
    import sys

    sys.path.append("./")

    from data.dbase.db_tables import Probe

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

    rsites = pd.DataFrame(
        (
            Probe.RecordingSite
            & 'mouse_id="BAA1110281"'
            & f'probe_configuration="longcol"'
        ).fetch()
    )

    render_probe_3d(rsites, targets=TARGETS)
    # render_probe_regions_slices(rsites, targets=TARGETS)
