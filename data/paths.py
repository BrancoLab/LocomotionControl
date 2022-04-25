from pathlib import Path
import sys

raw_data_folder = Path("W:\\swc\\branco\\Federico\\Locomotion\\raw")
local_raw_recordings_folder = Path(r"D:\recordings")
ccm_matrices = Path(r"W:\swc\branco\Federico\Locomotion\raw\CMM_matrices")
processed_tracking = Path(
    r"W:\swc\branco\Federico\Locomotion\processed\tracking"
)

probes_surgeries_metadata = Path(
    r"W:\swc\branco\Federico\Locomotion\raw\recordings_surgery_metadata.ods"
)


if sys.platform == "win32":
    analysis_folder = Path(
        r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis"
    )
else:
    analysis_folder = Path(
        "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis"
    )
