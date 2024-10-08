import configuration
import os
import re
import mne.io
import mne
import numpy as np


def load_configuration_parameters(subject, codename, mode):
    """
    """
    eeg_channel = configuration.EEG_CHANNEL
    path_file = os.path.join(configuration.DB_ROOT, subject + '.vhdr')
    scoring_path = os.path.join(configuration.ANNOTATIONS_ROOT, subject + '_scoring.txt')
    annotations_path = os.path.join(configuration.ANNOTATIONS_ROOT, subject + f"_annotations_{codename}_{mode}.txt")
    cut_off_freqs = configuration.CUT_OFF_FREQUENCIES

    return eeg_channel, path_file, scoring_path, annotations_path, cut_off_freqs


def detect_type_channel(ch_names):
    """
    Classify channels according their names in EOG, EEG and EMG
    
    Parameters
    ----------
    ch_names : list of str
        List containing channel names
    
    Returns
    ----------
    channels : dict
        Dictionary containing the classification of channels according their names
        > channels['eXg'] contains a list of eXg channels (X = e, o, m)
    """
    eog_names = '^(EOG|eog)'
    emg_names = '^(EMG|emg)'
    eeg_names = '^([AFfCcPpOoTtIiNn]|Fp)([1-9]|10|z)*'
    eeg = [ch for ch in ch_names if re.match(eeg_names, ch)]
    eog = [ch for ch in ch_names if re.match(eog_names, ch)]
    emg = [ch for ch in ch_names if re.match(emg_names, ch)]
    channels = dict(eeg=eeg, eog=eog, emg=emg)
    
    return channels


def load_file(file_path: str):
    """
    Import data into an MNE Raw object
    > Only format file allowed is vhdr from BrainVision

    Parameters
    ----------
    file_path : str (path-like)
        Path related to the position of the Brainvision file you want to read

    Returns
    ----------
    raw: raw.io.Raw
        Raw object from MNE containing the data
        > For more information see https://mne.tools/stable/generated/mne.io.Raw.html
    channels : dict
        Dictionary containing the classification of channels according their names
        > channels['eXg'] contains a list of eXg channels (X = e, o, m)
    """
    mne.set_log_level('CRITICAL')

    if os.path.splitext(file_path)[-1] == '.vhdr':
        raw0 = mne.io.read_raw_brainvision(file_path)
    else:
        raise ValueError('This program only works with Brainvision files')

    channels = detect_type_channel(raw0.ch_names)

    print('>> Names of channels detected')
    print('EEG:', channels['eeg'],
          '\nEMG:', channels['emg'],
          '\nEOG:', channels['eog'])
    print('\n>> All channels found in file:\n', 
          raw0.ch_names,'\n\n')

    raw = mne.io.read_raw_brainvision(file_path,
                    eog = channels['eog'],
                    misc = channels['emg'],
                    preload=True)

    emg_value = ['emg']*len(channels['emg'])
    dict_emg = {key: value for key, value in zip(channels['emg'], emg_value)}
    raw.set_channel_types(dict_emg)

    return raw, channels


def delete_duplicated_annotations(raw: mne.io.Raw):
    """
    Delete duplicated annotations in raw object
    """
    annots = raw.annotations
    new_annots_description = []
    new_annots_onset = []
    new_annots_duration = []
    for i, ann in enumerate(annots):
        if i == 0:
            new_annots_description.append(ann['description'])
            new_annots_onset.append(ann['onset'])
            new_annots_duration.append(ann['duration'])
        else:
            if ann['onset'] != annots[i-1]['onset']:
                new_annots_description.append(ann['description'])
                new_annots_onset.append(ann['onset'])
                new_annots_duration.append(ann['duration'])

    new_annotations = mne.Annotations(new_annots_onset, new_annots_duration, new_annots_description, orig_time=raw.info['meas_date'])
    raw.annotations.delete(np.arange(len(raw.annotations)))
    raw.set_annotations(new_annotations)
    return raw


def clean_annotations(raw: mne.io.Raw, thresholds: list):
    """
    """
    annots = raw.annotations
    new_annots_description = []
    new_annots_onset = []
    new_annots_duration = []
    for i, ann in enumerate(annots):
        if ann['duration']<=thresholds[1] and ann['duration']>=thresholds[0]:
            new_annots_description.append(ann['description'])
            new_annots_onset.append(ann['onset'])
            new_annots_duration.append(ann['duration'])
        else:
            print(f'Deleting annotation {ann["description"]} with duration {ann["duration"]} at {ann["onset"]}')

    new_annotations = mne.Annotations(new_annots_onset, new_annots_duration, new_annots_description, orig_time=raw.info['meas_date'])
    raw.annotations.delete(np.arange(len(raw.annotations)))
    raw.set_annotations(new_annotations)
    return raw