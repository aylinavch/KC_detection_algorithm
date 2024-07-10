import os
import sys
import configuration
from src.data.load_data import load_file, delete_duplicated_annotations
from src.data.preprocess import filter_raw_depending_on_channel_type, re_structure, set_sleep_stages, set_KC_labels, bandpass_filter
from src.features.build_features import plot_events, get_events, read_events, save_mean_figures, plot_3d_figures, save_characteristics_csv
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    """

    """
    if len(sys.argv) == 2:
        for subject in configuration.SUBJECTS:
            print(f'-------------------------------------{subject}-------------------------------------')
            file_path = os.path.join(configuration.DB_ROOT, subject+'.vhdr')
            
            if sys.argv[-1]=='-plot': 
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
                raw_SW_filtered = bandpass_filter(raw_with_KC, selected_channels=[eeg_channel], cut_off_frequencies=[0.5, 4])
                new_raw = delete_duplicated_annotations(raw_SW_filtered)
                reports_path = os.path.join(configuration.REPORTS_ROOT, 'figures')
                plot_events(new_raw, eeg_channel, subject, reports_path)

            elif sys.argv[-1]=='-get-events':
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
                reports_path = os.path.join(configuration.REPORTS_ROOT, 'npy')
                get_events(new_raw, eeg_channel, subject, reports_path, timelocked2='min')
                get_events(new_raw, eeg_channel, subject, reports_path, timelocked2='center')

    else:
        for i, subject in enumerate(configuration.SUBJECTS):
            
            file_path = os.path.join(configuration.DB_ROOT, subject+'.vhdr')
            raw, _ = load_file(file_path)
            sfreq = raw.info['sfreq']

            print(f'... Reading KC and noKC from {subject} ...')
            reports_subject_path = os.path.join(configuration.REPORTS_ROOT, 'npy', 'per_subject')
            if i == 0:
                all_KCs_timelocked2center, all_KCs_timelocked2min, all_noKCs = read_events(subject, reports_subject_path)
            else:
                KC_timelocked2center, KC_timelocked2min, noKC = read_events(subject, reports_subject_path)
                all_KCs_timelocked2center = np.vstack((all_KCs_timelocked2center, KC_timelocked2center))
                all_KCs_timelocked2min = np.vstack((all_KCs_timelocked2min, KC_timelocked2min))
                all_noKCs = np.vstack((all_noKCs, noKC))
        
        reports_path = os.path.join(configuration.REPORTS_ROOT, 'features')
        save_mean_figures(reports_path, all_KCs_timelocked2center, all_KCs_timelocked2min, all_noKCs)
        plot_3d_figures(reports_path, all_KCs_timelocked2center, all_KCs_timelocked2min, all_noKCs)
        save_characteristics_csv(reports_path, all_KCs_timelocked2center, all_KCs_timelocked2min, all_noKCs, sfreq)
            
            