from src.base.localizator import get_candidates
from src.data.load_data import load_configuration_parameters, load_file, delete_duplicated_annotations
from src.data.preprocess import filter_raw_depending_on_channel_type, structure_data_depending_on_channel_type, set_sleep_stages_labels, set_candidates_labels
from src.visualization.visualization import plot
from src.base.io import check_if_there_is_old_annotation_file, check_file_if_ready_to_save


def run_blind_labeling(subject: str, codename: str, prev_raw = None):
    
    mode = 'blind'
    eeg_channel, path_file, scoring_path, annotations_path, cut_off_freqs = load_configuration_parameters(subject, codename, mode)
     
    if not prev_raw:   
        #Load file
        raw, channels = load_file(path_file)

        #Filter each channel depending on type
        raw_filtered = filter_raw_depending_on_channel_type(raw, channels, cut_off_freqs)

        #Structure data depending on channel type
        raw_restructured, _ = structure_data_depending_on_channel_type(raw_filtered, channels, eeg_channels_selected=[eeg_channel])
        
        #Load and set labels (scoring and old annotations)
        raw_with_sleep_stages = set_sleep_stages_labels(raw_restructured, scoring_path)

        # Check if there is any old annotation file
        raw_checked = check_if_there_is_old_annotation_file(annotations_path, raw_with_sleep_stages)
    else:
        raw_checked = prev_raw
    
    #Plot signals and candidates to label events
    raw_plot = delete_duplicated_annotations(raw_checked)
    raw_cleaned = plot(raw_plot, title=f'{subject}')
    raw_new = delete_duplicated_annotations(raw_cleaned)

    #Save changes
    if not check_file_if_ready_to_save(raw_new, raw_plot, annotations_path, mode, codename, subject):
        run_blind_labeling(subject, codename, raw_new)


def run_semi_automatic_labeling(subject: str, codename: str, prev_raw = None):

    mode = 'semiauto'
    eeg_channel, path_file, scoring_path, annotations_path, cut_off_freqs = load_configuration_parameters(subject, codename, mode)
     
    if not prev_raw:   
        #Load file
        raw, channels = load_file(path_file)

        #Filter each channel depending on type
        raw_filtered = filter_raw_depending_on_channel_type(raw, channels, cut_off_freqs)

        #Structure data depending on channel type
        raw_restructured, _ = structure_data_depending_on_channel_type(raw_filtered, channels, eeg_channels_selected=[eeg_channel])
        
        #Load and set labels (scoring and old annotations)
        raw_with_sleep_stages = set_sleep_stages_labels(raw_restructured, scoring_path)
        
        # Get candidates
        eeg_signal = raw_with_sleep_stages.get_data(picks=[eeg_channel])[0]
        KC_candidates_onset_duration = get_candidates(eeg_signal, sfreq=raw.info['sfreq'], path_scoring=scoring_path, window_length=1)
        raw_with_candidates = set_candidates_labels(raw_with_sleep_stages, KC_candidates_onset_duration)
        
        # Check if there is any old annotation file
        raw_checked = check_if_there_is_old_annotation_file(annotations_path, raw_with_candidates)
    else:
        raw_checked = prev_raw
    
    #Plot signals and candidates to label events
    raw_plot = delete_duplicated_annotations(raw_checked)
    raw_cleaned = plot(raw_plot, title=f'{subject}')
    raw_new = delete_duplicated_annotations(raw_cleaned)

    #Save changes
    if not check_file_if_ready_to_save(raw_new, raw_plot, annotations_path, mode, codename, subject):
        run_semi_automatic_labeling(subject, codename, raw_new)
    
    
def run_automatic_labeling(subject: str, codename: str):
    mode = 'auto'
    print('-------------- Running automatic labeling --------------')
    print('Not implemented yet :)')