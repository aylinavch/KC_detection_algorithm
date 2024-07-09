import os
import sys
import configuration
from src.data.load_data import load_file, delete_duplicated_annotations, clean_annotations
from src.data.preprocess import filter_raw_depending_on_channel_type, add_channel_to_raw, set_sleep_stages, set_KC_labels, re_structure, get_only_KC_labels
from src.visualization.visualization import plot
from src.utils.localizator import get_flags, count_KC_noKC

if __name__ == '__main__':
    """
    Two modes:
    1) -labeling (default): Plot signal and clean/label K-Complexes (with localizator mode on)
    2) -cleaning: Clean K-Complexes (duration between 0.5s and 2s; no duplicated)
    """
    for subject in configuration.SUBJECTS:
        print(f'-------------------------------------{subject}-------------------------------------')
        file_path = os.path.join(configuration.DB_ROOT, subject+'.vhdr')
        
        if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[-1]=='-labeling'): # default
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
            signal = raw_with_KC.get_data(picks=[eeg_channel])[0]
            flags = get_flags(signal, sfreq=raw.info['sfreq'], path_scoring=scoring_path, window_length=1)
            raw_plot = add_channel_to_raw(raw_with_KC, channels_restructure, flags, 'LOC', 'eeg')

            #Plot signals and candidates to label KC
            new_raw = delete_duplicated_annotations(raw_plot)
            raw_cleaned = plot(new_raw, title=f'{subject}')
            #new_raw = delete_duplicated_annotations(raw_cleaned)

            #Save changes
            if raw_cleaned.annotations != raw_with_KC.annotations:
                raw_only_KC = get_only_KC_labels(raw_cleaned)
                try:
                    os.rename(KC_path, os.path.join(configuration.ANNOTATIONS_ROOT, subject+'_old_annotations.txt'))
                except FileNotFoundError:
                    pass
                raw_only_KC.annotations.save(os.path.join(configuration.ANNOTATIONS_ROOT, subject+'_annotations.txt'))
            else:
                print('No changes')
            
            count_KC_noKC(raw_cleaned)
        
        elif len(sys.argv) == 2 and sys.argv[-1]=='-semiautomatic':
            print('Not developed yet')

        elif len(sys.argv) == 2 and sys.argv[-1]=='-cleaning':
            raw, channels = load_file(file_path)

            scoring_path = os.path.join(configuration.ANNOTATIONS_ROOT, subject+'_scoring.txt')
            raw_annotated = set_sleep_stages(raw, scoring_path)
            KC_path = os.path.join(configuration.ANNOTATIONS_ROOT, subject+'_annotations.txt')
            raw_with_KC = set_KC_labels(raw_annotated, KC_path)

            new_raw = delete_duplicated_annotations(raw_with_KC)

            #Save changes
            if new_raw.annotations != raw.annotations:
                raw_only_KC = get_only_KC_labels(new_raw)
                raw_KC_cleaned = clean_annotations(raw_only_KC)
                raw_KC_cleaned.annotations.save(os.path.join(configuration.ANNOTATIONS_ROOT, subject+'_annotations_cleaned.txt'), overwrite=True)
            else:
                print('No duplicated found')
            