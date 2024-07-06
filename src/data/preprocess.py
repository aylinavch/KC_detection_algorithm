import re
import numpy as np
import mne.io
import numpy as np
from scipy.signal import square

def pulse(N, sfreq):
    """
    Create artificial signal with a 0.5 sec pulse to do the grid on the interface
    """
    t = np.linspace(0, round(N/sfreq), N, endpoint=False) 
    signal_pulse = square(2 * np.pi * 1 * t)
    return signal_pulse


def re_structure(raw: mne.io.Raw, channels: list, eeg_channels_selected=['C3_1','C4_1'], plotting=True):
    """
    """
    eog, _ = raw.get_data(picks=channels['eog'])
    eeg = raw.get_data(picks=eeg_channels_selected)
    emg, _ = raw.get_data(picks=channels['emg'])

    if len(eog) == 2:
        eog = eog[0]
    if len(emg) == 2:
        emg = emg[1] - emg[0] 
    
    num_of_channels = 2 + len(eeg_channels_selected)
    new_data = np.empty((num_of_channels, raw.n_times))
    new_data[0] = eog
    for i in range(len(eeg_channels_selected)):
        new_data[1+i] = eeg[i]
    new_data[-1] = emg

    new_ch_names = ['EOG'] + eeg_channels_selected + ['EMG']
    new_ch_types = ['eog'] + len(eeg_channels_selected)*['eeg'] + ['emg']
    new_channels = {'eeg': eeg_channels_selected, 
                    'eog': ['EOG'], 
                    'emg': ['EMG']}

    if plotting:
        new_data = np.insert(new_data, 2, pulse(raw.n_times, raw.info['sfreq']), axis=0)
        new_ch_names.insert(2, 'grid')
        new_ch_types.insert(2, 'misc')
        num_of_channels+=1
    
    new_info = mne.create_info(new_ch_names, sfreq=raw.info['sfreq'], ch_types=new_ch_types)
    new_info.set_meas_date(raw.info['meas_date'])
    new_raw = mne.io.RawArray(new_data, new_info)
    new_raw.set_annotations(raw.annotations)

    return new_raw, new_channels

def get_only_KC_labels(raw: mne.io.Raw):
    """
    """
    regex_KC = r"^KC(?:_\w+)?$"
    regex_noKC = r"^noKC(?:_\w+)?$"

    all_annotations = raw.annotations

    KC_onset = [ann['onset'] for ann in all_annotations if re.match(regex_KC, ann['description'])]
    KC_duration = [ann['duration'] for ann in all_annotations if  re.match(regex_KC, ann['description'])]
    KC_description = [ann['description'] for ann in all_annotations if re.match(regex_KC, ann['description'])]
    KC_annotations = mne.Annotations(KC_onset, KC_duration, KC_description, orig_time=raw.info['meas_date'])

    noKC_onset = [ann['onset'] for ann in all_annotations if re.match(regex_noKC, ann['description'])]
    noKC_duration = [ann['duration'] for ann in all_annotations if  re.match(regex_noKC, ann['description'])]
    noKC_description = [ann['description'] for ann in all_annotations if re.match(regex_noKC, ann['description'])]
    noKC_annotations = mne.Annotations(noKC_onset, noKC_duration, noKC_description, orig_time=raw.info['meas_date'])

    only_KC_annotations = KC_annotations + noKC_annotations
    
    raw_only_KC = raw.copy().set_annotations(only_KC_annotations)

    return raw_only_KC


def set_KC_labels(raw: mne.io.Raw, KC_path: str):
    """
    ...

    Parameters
    ----------
    raw : raw.io.Raw
        Raw object from MNE containing the data

    KC_path: str (path-like)
        Path related to the labeling txt file with KC you want to read
   
    Returns
    ----------
    raw_labeled : raw.io.Raw
        Raw object from MNE containing the scoring annotations
    """
    regex_KC = r"^KC(?:_\w+)?$"
    regex_noKC = r"^noKC(?:_\w+)?$"

    try:
        old_annots = mne.read_annotations(KC_path, sfreq=raw.info['sfreq'])

        KC_onset = [ann['onset'] for ann in old_annots if re.match(regex_KC, ann['description'])]
        KC_duration = [ann['duration'] for ann in old_annots if  re.match(regex_KC, ann['description'])]
        KC_description = [ann['description'] for ann in old_annots if re.match(regex_KC, ann['description'])]
        KC_annotations = mne.Annotations(KC_onset, KC_duration, KC_description, orig_time=raw.info['meas_date'])

        noKC_onset = [ann['onset'] for ann in old_annots if re.match(regex_noKC, ann['description'])]
        noKC_duration = [ann['duration'] for ann in old_annots if  re.match(regex_noKC, ann['description'])]
        noKC_description = [ann['description'] for ann in old_annots if re.match(regex_noKC, ann['description'])]
        noKC_annotations = mne.Annotations(noKC_onset, noKC_duration, noKC_description, orig_time=raw.info['meas_date'])

        all_annotations = old_annots + KC_annotations + noKC_annotations
        
        raw_labeled = raw.copy().set_annotations(all_annotations)

        return raw_labeled
    
    except FileNotFoundError: #There is no KC file
        return raw.copy()


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