import shutil
from pathlib import Path
from rich.prompt import Confirm
from rich.progress import track
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ! This is to clean up data produced by CONTROL before running RNN


fld = Path(
    "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\RNN\\training_data_fixed_dur"
)
subflds = [f for f in fld.glob("*") if f.is_dir()]

to_discard = []

for n, sub in track(enumerate(subflds), total=len(subflds)):
    try:
        img = mpimg.imread(sub / "outcome.png")
    except FileNotFoundError:
        continue

    f, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(img)
    plt.show()

    if not Confirm.ask(
        f"Do you want to keep simulation {n} of {len(subflds)}?", default=True
    ):
        shutil.rmtree(sub)
