from fcutils.path import from_json
from pathlib import Path

main_folder = Path(
    "Z:\\swc\\branco\\Federico\\Locomotion\\control\\behavioural_data"
)
tracking_folder = main_folder / "tracking"

metadata_path = main_folder / "metadata.json"
videos_metadata_path = main_folder / "videos_metadata.json"

bash_scripts_folder = main_folder / "bash_scripts"

# get subfolders
metadata = from_json(metadata_path)

subfolders = {exp: main_folder / exp for exp in metadata.keys()}
