import re
import numpy as np
import mne.io
import numpy as np
from src.utils.sleep_stages_utils import get_scoring_from_path

def re_structure(raw: mne.io.Raw, channels: list, eeg_channels_selected=['C3_1','C4_1']):
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
        old_annots = mne.read_annotations(KC_path, sfreq=raw.info['sfreq']) + raw.annotations

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
    stages = get_scoring_from_path(path_scoring)
    num_of_epochs = stages.shape[0]
    
    assert num_of_epochs == len(raw)//(raw.info['sfreq']*epoch_duration), "File with sleep stages annotations has a different amount of annotations comparing to the recording length"

    onset = np.zeros((num_of_epochs))        
    duration = np.zeros((num_of_epochs))    
    description = np.zeros((num_of_epochs))

    start = 0
    for i in range(num_of_epochs):
        onset[i] = start
        duration[i] = epoch_duration 
        description[i] = stages[i]
        start = start + epoch_duration
    
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


def filter_raw_depending_on_channel_type(raw, channels, cut_off_freqs):
    """
    Filter each channel depending on type (EEG, EOG and EMG)

    Parameters
    ----------
    raw : raw.io.Raw
        Raw object from MNE containing the data

    channels: list
        List containing the name of channels to bandpass filter
    
    cut_off_freqs: dict
        Dictionary with the cut off frequencies for each channel type
    
    Returns
    ----------
    raw_filtered : raw.io.Raw
        Raw object from MNE containing the data filtered
    """    
    raw_filtered = bandpass_filter(raw.copy(), channels['eeg'], cut_off_freqs['eeg'])
    raw_filtered = bandpass_filter(raw_filtered, channels['emg'], cut_off_freqs['emg'])
    raw_filtered = bandpass_filter(raw_filtered, channels['eog'], cut_off_freqs['eog'])

    return raw_filtered

def add_channel_to_raw(raw, channels, new_channel, name='LOC', type_ch='eeg'):
    """
    """
    eog = raw.get_data(picks=channels['eog'])
    eeg = raw.get_data(picks=channels['eeg'])
    emg = raw.get_data(picks=channels['emg'])

    if len(eog) == 2:
        eog = eog[0]
    if len(emg) == 2:
        emg = emg[1] - emg[0] 
    
    num_of_channels = 2 + len(channels['eeg']) + 1
    new_data = np.empty((num_of_channels, raw.n_times))
    new_data[0] = eog
    for i in range(len(channels['eeg'])):
        new_data[1+i] = eeg[i]
    new_data[-2] = new_channel
    new_data[-1] = emg

    new_ch_names = ['EOG'] + channels['eeg'] + [name] + ['EMG']
    new_ch_types = raw.get_channel_types()
    new_ch_types.insert(num_of_channels-2, type_ch)

    new_info = mne.create_info(new_ch_names, sfreq=raw.info['sfreq'], ch_types=new_ch_types)
    new_info.set_meas_date(raw.info['meas_date'])
    new_raw = mne.io.RawArray(new_data, new_info)
    new_raw.set_annotations(raw.annotations)

    return new_raw
