import numpy as np


def get_scoring_from_path(path_scoring):

    return np.loadtxt(path_scoring, delimiter =' ', usecols =(0) )


def set_sleep_stages_per_sample(signal, path_scoring: str, epoch_duration: int =30, sfreq: int =250):
    """
    """
    stages = get_scoring_from_path(path_scoring)
    assert len(stages) == len(signal)//(epoch_duration*sfreq), "File with sleep stages annotations has a different amount of annotations comparing to the recording length"
    
    stages_per_sample = np.repeat(stages, epoch_duration*sfreq, axis=0) 

    return stages_per_sample