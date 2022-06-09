from loguru import logger
import pandas as pd
from pathlib import Path

from fcutils.path import from_yaml, files
from fcutils.progress import track

from data.dbase._tables import insert_entry_in_table
from data.paths import raw_data_folder


def fill_session_table(table):
    logger.info("Filling in session table")
    in_table = table.fetch("name")
    mice = from_yaml("data\dbase\mice.yaml")

    # Load recordings sessoins metadata
    recorded_sessions = pd.read_excel(
        table.recordings_metadata_path, engine="odf"
    )

    # Get the videos of all sessions
    # vids = [f for f in files(raw_data_folder / "video") if ".avi" in f.name]
    vids = [f for f in files(Path(r"K:\analog_inputs_temp"), "*.BIN")]

    for video in track(vids, description="Adding sessions", transient=True):
        # Get session data
        # name = video.name.split("_video")[0]
        name = video.name.split("_analog")[0]
        if name in in_table:
            continue

        if "opto" in name:
            logger.info(
                f"Skipping session {name} because its OPTOGENETICS session"
            )
            continue

        if "test" in name.lower() in name.lower():
            logger.info(f"Skipping session {name} as it is a test")
            continue

        # get date and mouse
        try:
            date = name.split("_")[1]
            mouse = [
                m["mouse"]["mouse_id"]
                for m in mice
                if m["mouse"]["mouse_id"] in name
            ][0]
        except IndexError:
            if "523" in name:
                date = name.split("_")[1]
                mouse = "BAA110523"
            elif "522" in name:
                date = name.split("_")[1]
                mouse = "BAA110522"
            elif "521" in name:
                date = name.split("_")[1]
                mouse = "BAA110521"
            elif "520" in name:
                date = name.split("_")[1]
                mouse = "BAA110520"
            else:
                logger.warning(
                    f"Skipping session {name} because couldnt figure out the mouse or date it was done on"
                )
            continue
        key = dict(mouse_id=mouse, name=name, date=date)

        # get file paths
        key["video_file_path"] = (
            raw_data_folder / "video" / (name + "_video.avi")
        )
        key["ai_file_path"] = (
            raw_data_folder / "analog_inputs" / (name + "_analog.bin")
        )

        key["csv_file_path"] = (
            raw_data_folder / "analog_inputs" / (name + "_data.csv")
        )

        # if (
        #     not key["video_file_path"].exists()
        #     or not key["ai_file_path"].exists()
        # ):
        #     raise FileNotFoundError(
        #         f"Either video or AI files not found for session: {name} with data:\n{key}"
        #     )

        # get ephys files & arena type
        if name in recorded_sessions["bonsai filename"].values:
            rec = recorded_sessions.loc[
                recorded_sessions["bonsai filename"] == name
            ].iloc[0]
            base_path = (
                table.recordings_raw_data_path
                / rec["recording folder"]
                / (rec["recording folder"] + "_imec0")
                / (rec["recording folder"] + "_t0.imec0")
            )
            key["ephys_ap_data_path"] = str(base_path) + ".ap.bin"
            key["ephys_ap_meta_path"] = str(base_path) + ".ap.meta"

            key["arena"] = rec.arena
            key["is_recording"] = 1
            key["date"] = int(rec.date)
        else:
            key["ephys_ap_data_path"] = ""
            key["ephys_ap_meta_path"] = ""
            key["arena"] = "hairpin"
            key["is_recording"] = 0

        # add to table
        # if "." in str(key["date"]):
        #     a = 1
        insert_entry_in_table(key["name"], "name", key, table)
