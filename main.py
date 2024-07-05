import os
import sys
import configuration
from src.data.load_data import load_file
from src.data.preprocess import bandpass_filter, set_sleep_stages, set_KC_labels, re_structure
from src.visualization.visualization import plot
from src.models.KC_detector_model import semiautomatic_detection

if __name__ == '__main__':
    """
    Three modes:
    1) -labeling (default): Plot signal and clean/label K-Complexes
    2)
    3)
    """
    for subject in configuration.SUBJECTS:
        print(f'-------------------------------------{subject}-------------------------------------')
        file_path = os.path.join(configuration.DB_ROOT, subject+'.vhdr')
        
        if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[-1]=='-labeling'): # default
            #Load file
            raw, channels = load_file(file_path)

            #Filter each channel depending on type
            raw_filtered = bandpass_filter(raw, channels['eeg'], [0.16, 35])
            raw_filtered = bandpass_filter(raw_filtered, channels['emg'], [10, 90])
            raw_filtered = bandpass_filter(raw_filtered, channels['eog'], [0.16, 10])

            #Restructure data
            raw_restructure, _ = re_structure(raw_filtered, channels, eeg_channels_selected=['C3_1','C4_1'], plotting=True)
            
            #Load and set labels (scoring and KCs)
            scoring_path = os.path.join(configuration.ANNOTATIONS_ROOT, subject+'_scoring.txt')
            raw_annotated = set_sleep_stages(raw_restructure, scoring_path)
            KC_path = os.path.join(configuration.ANNOTATIONS_ROOT, subject+'_KCs_and_scoring.txt')
            raw_with_KC = set_KC_labels(raw_annotated, KC_path)

            #Plot signals and label KC
            raw_cleaned = plot(raw_with_KC.copy(), title=f'{subject}')

            #Save changes
            if raw_cleaned.annotations != raw_with_KC.annotations:
                raw_cleaned.annotations.save(os.path.join(configuration.ANNOTATIONS_ROOT, subject+'_KCs_and_scoring.txt'))
            else:
                print('No changes')
        
        
        elif len(sys.argv) == 2 and sys.argv[-1]=='-semiautomatic':
            #Load file
            raw, channels = load_file(file_path)

            #Filter each channel depending on type
            raw_filtered = bandpass_filter(raw, channels['eeg'], [0.16, 35])
            raw_filtered = bandpass_filter(raw_filtered, channels['emg'], [10, 90])
            raw_filtered = bandpass_filter(raw_filtered, channels['eog'], [0.16, 10])

            #Restructure data
            raw_restructure, _ = re_structure(raw_filtered, channels, eeg_channels_selected=['C3_1','C4_1'], plotting=True)
            
            #Semiautomatic detection
            semiautomatic_detection(raw_restructure, eeg_channels_selected=['C4_1'])
            