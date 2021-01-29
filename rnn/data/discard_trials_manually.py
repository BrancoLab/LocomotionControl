# %%
from pigeon import annotate
from IPython.display import display, Image
from pathlib import Path

from pyinspect.utils import subdirs, dir_files

# ! This is to clean up data produced by CONTROL before running RNN


fld = Path("D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\RNN\\training_data")

# get paths to all images
flds = subdirs(fld)
images = {
    fld: dir_files(fld, "outcome.png")[0]
    for fld in flds
    if dir_files(fld, "outcome.png")
}
lookup = {v: k for k, v in images.items()}


# %%
# do a first round of manual selection
annotations = annotate(
    images.values(),
    options=["good", "bad"],
    display_fn=lambda filename: display(Image(filename)),
)

for fpath, result in annotations:
    if not result == "good":
        fpath.unlink()

kept = {k: v for k, v in images.items() if v.exists()}
print(f"Kept {len(kept)}/{len(images)} simulations")
# %%
# second round of quality checks
annotations = annotate(
    kept.values(),
    options=["good", "bad"],
    display_fn=lambda filename: display(Image(filename)),
)

for fpath, result in annotations:
    if not result == "good":
        fpath.unlink()

kept = {k: v for k, v in images.items() if v.exists()}
print(f"Kept {len(kept)}/{len(images)} simulations")
# %%
