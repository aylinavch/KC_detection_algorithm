import numpy as np
import mne.io

def set_sleep_stages(raw: mne.io.Raw, path_scoring: str, epoch_duration = 30):
    """
    Set sleep stages annotations to raw object

    Parameters
    ----------
    raw : raw.io.Raw
        Raw object from MNE containing the data

    path_scoring: str (path-like)
        Path related to the position of the scoring txt file you want to read

    epoch_duration: int
        Duration of each epoch taken into account to do the scoring (in seconds)
    
    Returns
    ----------
    raw_scored : raw.io.Raw
        Raw object from MNE containing the scoring annotations
    """
    stages = np.loadtxt(path_scoring, delimiter =' ', usecols =(0) )
    num_of_epochs = stages.shape[0]
    epoch_length = 30
    
    assert num_of_epochs == len(raw)/raw.info['sfreq']//epoch_duration, "File with sleep stages annotations has a different amount of annotations comparing to the recording length"

    onset = np.zeros((num_of_epochs))        
    duration = np.zeros((num_of_epochs))    
    description = np.zeros((num_of_epochs))

    start = 0
    for i in range(num_of_epochs):
        onset[i] = start
        duration[i] = epoch_length 
        description[i] = stages[i]
        start = start + epoch_length
    
    stages_anot = mne.Annotations(onset, duration, description, orig_time=raw.info['meas_date'])    
    raw_scored = raw.copy().set_annotations(stages_anot)

    return raw_scored


def bandpass_filter(raw: mne.io.Raw, selected_channels: list, cut_off_frequencies: list):
    """
    Bandpass filter applied to specific channels indicating which cut off frequencies.

    Parameters
    ----------
    raw : raw.io.Raw
        Raw object from MNE containing the data

    selected_channels: list
        List containing the name of channels to bandpass filter

    cut_off_frequencies: list
        List with two elements, the first one is the cut off frequency of the highpass filter
        The second element is the cut off frequency of the lowpass filter
    
    Returns
    ----------
    raw_filtered : raw.io.Raw
        Raw object from MNE containing the data filtered
    """
    assert len(cut_off_frequencies) == 2, f"You must specify two elements in the list, {len(cut_off_frequencies)} were given"

    raw_filtered = raw.copy().filter(
        l_freq=cut_off_frequencies[0], h_freq=cut_off_frequencies[1],
        picks=selected_channels)

    return raw_filtered