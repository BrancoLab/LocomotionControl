
from loguru import logger

from fcutils.maths.signals import get_onset_offset

from data.dbase.io import load_bin

def get_triggers(session: dict, sampling_rate:int=30000) -> dict:
    name = session['name']
    logger.debug(f'Getting bonsai trigger times for session "{name}"')

    # load bin data
    analog = load_bin(session['ai_file_path'], nsigs=session['n_analog_channels'])

    # check that the number of frames is correct
    frames_onsets, frames_offsets = get_onset_offset(analog[:, 0], 2.5)


    # check everything correct
    if len(frames_onsets) != len(frames_offsets):
        raise ValueError('Mismatch between number of frames onsets and offsets')
    if len(frames_onsets) != session['n_frames']: 
        raise ValueError('Mismatch between frame onsets and expected number of frames')


    # align time stamps to bonsai cut sample
    frames_onsets -= session['bonsai_cut_start']
    frames_offsets -= session['bonsai_cut_start']

    # get duration
    n_samples = frames_offsets[-1] - frames_onsets[0]
    duration_ms = n_samples /sampling_rate / 1000


    # return results
    return dict(trigger_times=frames_onsets, n_samples=n_samples, n_ms=duration_ms)

