from brainrender import Scene
from brainrender.video import VideoMaker


from rich import print
from myterial import orange
from pathlib import Path

print(f"[{orange}]Running example: {Path(__file__).name}")

# Create a scene
scene = Scene("my video")
scene.root.alpha(1)

# Create an instance of video maker
vm = VideoMaker(
    scene,
    r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\ephys",
    "nunits vid",
)

# make a video with the custom make frame function
vm.make_video(
    azimuth=2, duration=3, fps=15, render_kwargs=dict(camera="sagittal")
)
