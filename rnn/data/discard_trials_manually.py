import shutil
from pathlib import Path
from rich.prompt import Confirm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ! This is to clean up data produced by CONTROL before running RNN


fld = Path(
    "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/control/RNN/simulated_training_data"
)
subflds = [f for f in fld.glob("*") if f.is_dir()]

to_discard = []

for n, sub in enumerate(subflds):
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
