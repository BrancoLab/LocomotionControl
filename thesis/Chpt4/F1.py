from brainrender import Scene, settings
from brainrender.atlas_specific import get_streamlines_for_region
from brainrender.actors.streamlines import make_streamlines

from rich import print
from myterial import orange
from pathlib import Path
from myterial import blue_grey_darker

settings.SHOW_AXES = False
settings.vsettings.screenshotTransparentBackground = True


print(f"[{orange}]Running example: {Path(__file__).name}")

# ---------------------------------------------------------------------------- #
#                                      RSP                                     #
# ---------------------------------------------------------------------------- #

# # Create a brainrender scene
# scene = Scene(screenshots_folder="D:\Dropbox (UCL)\Rotation_vte\Writings\THESIS\Chpt4\Plots")

# # Add brain regions
# regions = scene.add_brain_region("MOs", "RSP", alpha=0.5)
# scene.slice("sagittal", actors=regions)


# # Get stramlines data and add
# streams = get_streamlines_for_region("RSP")[:2]
# streams = scene.add(*make_streamlines(*streams, color=blue_grey_darker, alpha=1))

# # Render!
# scene.render(camera="sagittal2", interactive=False)
# scene.screenshot("RSP_streamlines_sagittal.png")
# scene.close()


# # Create a brainrender scene
# scene = Scene(screenshots_folder="D:\Dropbox (UCL)\Rotation_vte\Writings\THESIS\Chpt4\Plots")

# # Add brain regions
# regions = scene.add_brain_region("MOs", "RSP", alpha=0.5)

# # Get stramlines data and add
# streams = get_streamlines_for_region("RSP")[:2]
# scene.add(*make_streamlines(*streams, color=blue_grey_darker, alpha=1))

# # Render!
# scene.render(camera="top", zoom=1, interactive=False)
# scene.screenshot("RSP_streamlines_top.png")
# scene.close()


# ---------------------------------------------------------------------------- #
#                                      MOs                                     #
# ---------------------------------------------------------------------------- #

# # Create a brainrender scene
# scene = Scene(screenshots_folder="D:\Dropbox (UCL)\Rotation_vte\Writings\THESIS\Chpt4\Plots")

# # Add brain regions
# regions = scene.add_brain_region("MOs", "CUN", "PPN", "GRN", alpha=0.5)


# # Get stramlines data and add
# streams = get_streamlines_for_region("MOs")[:2]
# scene.add(*make_streamlines(*streams, color=blue_grey_darker, alpha=1))

# # Render!
# scene.render(camera="sagittal2", zoom=1, interactive=False)
# scene.screenshot("MOs_streamlines_sagittal.png")
# scene.close()


# Create a brainrender scene
scene = Scene(
    screenshots_folder="D:\Dropbox (UCL)\Rotation_vte\Writings\THESIS\Chpt4\Plots"
)

# Add brain regions
regions = scene.add_brain_region("MOs", "CUN", "PPN", "GRN", alpha=0.5)


# Get stramlines data and add
streams = get_streamlines_for_region("MOs")[:2]
scene.add(*make_streamlines(*streams, color=blue_grey_darker, alpha=1))

# Render!
scene.render(camera="top", zoom=1, interactive=False)
scene.screenshot("MOs_streamlines_top.png")
scene.close()
