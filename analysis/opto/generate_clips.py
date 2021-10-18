import sys
sys.path.append("./")

import cv2
import numpy as np

import pandas as pd
from loguru import logger
from fcutils import video

"""
    Generates videos where for each ROI of the opto experiments
    it extracts all the stim times and creates videos of what's happening

    ! CURRENTLY THE ROI INFORMATION ISNOT EXTRACTED !
"""

from data.dbase.db_tables import (
    Session,
    OptoImplant,
    OptoSession,
    OptoStimuli,
)

def make(sessions: pd.DataFrame, save_path:str, n_frames_before:int, n_frames_after:int) -> None:
    """
        It takes all the optogenetics stim times for mice implanted to target
        a given region and with the stimulation happening for a given ROI.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # iterate over experimental sessions
    for i, session in sessions.iterrows():
        nframes, width, height, fps, is_color = video.get_video_params(session.video_file_path)

        # create an opencv writer
        if i == 0:
            writer = video.open_cvwriter(
                save_path,
                w=width,
                h=height,
                framerate=int(fps),
                format=".mp4",
                iscolor=is_color,
            )

        # iterate over trigger times
        for trigger in session.stim_onsets:
            start = trigger - n_frames_before
            end = trigger + n_frames_after
            
            # iterate over frames
            video.cap_set_frame(video, start)
            for framen in range(start, end):
                ret, frame = video.read()
                if not ret:
                    logger.debug(
                        f"FCUTILS: failed to read frame {framen} while trimming clip [{nframes} frames long]."
                    )
                    break

                # add marker to show stim is on
                frame = cv2.circle(frame, (50, 50), 25, (0, 0, 0), -1)
                if framen >= trigger:
                    frame = cv2.circle(frame, (50, 50), 20, (255, 255, 255), -1)

                # add text with metadata
                cv2.putText(frame, f'{session["name"]}: {trigger}', (10, height - 20), font, 3, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f'{session["name"]}: {trigger}', (10, height - 20), font, 3, (255, 255, 255), 2, cv2.LINE_AA)

                # write
                writer.write(frame)

            # add some black frames
            for i in range(10):
                writer.write(np.zeros_like(frame))

    writer.release()



def main():
    '''
        Get main metadata and call multiple make in parallel
    '''
    sessions = pd.DataFrame(
        (Session * OptoImplant * OptoSession * OptoStimuli).fetch()
    )

    # TODO filter session based on targets / ROIs
    # TODO and launch stuff in parallel

    save_path = str(r'D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\OPTO\opto_clips.mp4')
    make(sessions, save_path, n_frames_before=120, n_frames_after=180)



if __name__ == '__main__':
    main()