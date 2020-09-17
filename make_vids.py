# %%
from proj.animation.animate import animate_from_images
from pathlib import Path

# %%
# Get folders
fld = Path(
    "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/control"
)


flds = [f for f in fld.glob("*") if f.is_dir() and "tracking_" in f.name]
flds

# %%
for fld in flds:
    try:
        animate_from_images(
            str(fld / "frames"), str(fld.parent / fld.name) + "_vid.mp4"
        )
    except ValueError as e:
        print(e)

# %%
