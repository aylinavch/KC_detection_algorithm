import os
import sys
import configuration
from src.data.load_data import load_file, delete_duplicated_annotations
from src.data.preprocess import filter_raw_depending_on_channel_type, set_sleep_stages, set_KC_labels, re_structure
from src.features.build_features import get_events

if __name__ == '__main__':
    """
    """
    for subject in configuration.SUBJECTS:
        print(f'-------------------------------------{subject}-------------------------------------')
        file_path = os.path.join(configuration.DB_ROOT, subject+'.vhdr')
        eeg_channel = 'C4_1'
        #Load file
        raw, channels = load_file(file_path)

        #Filter each channel depending on type
        cut_off_freqs = {'eeg': [0.16, 35], 'emg': [10, 90], 'eog': [0.16, 10]}
        raw_filtered = filter_raw_depending_on_channel_type(raw, channels, cut_off_freqs)

        #Restructure data
        raw_restructure, channels_restructure = re_structure(raw_filtered, channels, eeg_channels_selected=[eeg_channel])
        
        #Load and set labels (scoring and KCs)
        scoring_path = os.path.join(configuration.ANNOTATIONS_ROOT, subject+'_scoring.txt')
        raw_annotated = set_sleep_stages(raw_restructure, scoring_path)
        KC_path = os.path.join(configuration.ANNOTATIONS_ROOT, subject+'_annotations.txt')
        raw_with_KC = set_KC_labels(raw_annotated, KC_path)

        # Get candidates to label KC
        new_raw = delete_duplicated_annotations(raw_with_KC)
        reports_path = os.path.join(configuration.REPORTS_ROOT, 'npy', 'characterization')
        get_events(new_raw, eeg_channel, subject, reports_path, timelocked2='center', window=3)
        get_events(new_raw, eeg_channel, subject, reports_path, timelocked2='center', window=3, just_get_new_start=True)
    