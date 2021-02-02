from loguru import logger
from pathlib import Path

from pyinspect.utils import dir_files
from fcutils.path import from_json, to_json

"""
    code to facilitate the manual selection of which trials should be kept for analysis

    Trials to analyze are based on:
        - mouse must escape
        - mouse must reach the shelter without stopping during escape
"""

folder = Path(
    "Z:\\swc\\branco\\Federico\\Locomotion\\control\\experimental_validation\\2WDD"
)
trials_folder = folder / "TRIALS_CLIPS"

records_path = folder / "trials_records.json"
if not records_path.exists():
    to_json(records_path, {})

records = from_json(records_path)

# get trials names
trials = dir_files(trials_folder, "*_trial_*.mp4")
logger.info(f"Found {len(trials)} in folder")

# fill in missing entries in records
for trial in trials:
    name = trial.stem
    if name not in records.keys():
        records[name] = dict(good="tbd")
to_json(records_path, records)


# check how many good trials we have
good, bad = 0, 0
for trial in records.values():
    if isinstance(trial["good"], str):
        continue
    if trial["good"]:
        good += 1
    else:
        bad += 1

logger.info(
    f"{round((good/(good+bad)) * 100, 2)}% of trials are good [{good}/{good + bad}]"
)
