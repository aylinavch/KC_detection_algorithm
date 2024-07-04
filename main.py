import os
import configuration
from src.data.load_data import load_file
from src.data.preprocess import bandpass_filter, set_sleep_stages

if __name__ == '__main__':
    for subject in configuration.SUBJECTS:
        print(f'-------------------------------------{subject}-------------------------------------')
        file_path = os.path.join(configuration.DB_ROOT, subject+'.vhdr')

        #Load file
        raw, channels = load_file(file_path)

        #Filter each channel depending on type
        raw_filtered = bandpass_filter(raw, channels['eeg'], [0.16, 35])
        raw_filtered = bandpass_filter(raw_filtered, channels['emg'], [10, 90])
        raw_filtered = bandpass_filter(raw_filtered, channels['eog'], [0.16, 10])

        #Load and set labels (scoring and KCs)
        scoring_path = os.path.join(configuration.SCORING_ROOT, subject+'.txt')
        raw_annotated = set_sleep_stages(raw_filtered, scoring_path)
        #KC_labeled = get_KCs_list(filtered_signal, name_channel='C4', sf = 250, pathKC = path_KC1)

        print('... Finished filtering and starting  ...')