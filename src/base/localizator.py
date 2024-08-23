import re
import os
import configuration
import joblib
import scipy as sp
import numpy as np
import mne.io
from src.base.sleep_stages_utils import set_sleep_stages_per_sample
from src.base.localizator_utils import get_zc_nearest_to
from matplotlib import pyplot as plt


def count_zero_crossings(arr: np.ndarray) -> int:
    num_zc = 0
    for i in range(1, len(arr)):
        if (arr[i-1] > 0 and arr[i] < 0) or (arr[i-1] < 0 and arr[i] > 0):
            num_zc += 1
    return num_zc


def check_if_meet_main_conditions(window, sfreq: int):
    """
    """
    maxi = max(window)
    mini = min(window)
    pos_maxi = np.where(window==maxi)[0][0] 
    pos_mini = np.where(window==mini)[0][0]

    p2p_amplitude = (maxi - mini)*1e6 #in microvolts
    dur_mini_maxi = (pos_maxi - pos_mini)/sfreq #in seconds

    if (pos_mini<pos_maxi) and (p2p_amplitude>=75) and (dur_mini_maxi<1):
        return True
    else:
        return False


def get_candidates(signal:np.ndarray, sfreq: int, path_scoring: str, window_length: int =1, stages_allowed:list =[2.0], step:float = 0.1):
    """
    """
    stages_per_sample = set_sleep_stages_per_sample(signal, path_scoring, epoch_duration=30, sfreq=sfreq)    
    signal = signal[:len(stages_per_sample)]
    pos_candidate = []
    pos = 0
    while pos < len(signal):
        if stages_per_sample[pos] in stages_allowed:
            window = signal[pos:pos+int(window_length*sfreq)]
            if check_if_meet_main_conditions(window, sfreq):
                try:
                    window, idx_start, _, _, _, idx_end, new_pos = detect_points_of_event(window, sfreq, signal, pos, window_length=3) 
                except AssertionError: # Could not find the points of the event correctly
                    pos += int(window_length/2*sfreq)
                    continue
                pos_candidate.append([(new_pos + idx_start)/sfreq, (idx_end-idx_start)/sfreq]) #[onset, duration]
                pos = new_pos + idx_end + 1
            else:
                pos += int(step*sfreq)
        else:
            pos += int(30*sfreq)
    
    predicted = []
    for p in pos_candidate:
        predicted.append([p[0],p[1]])
        
    return predicted


def count_KC_noKC(raw: mne.io.Raw):
    """
    """
    annots = raw.annotations
    regex_KC = r"^KC(?:_\w+)?$"
    regex_noKC = r"^noKC(?:_\w+)?$"
    KC = [ann['description'] for ann in annots if re.match(regex_KC, ann['description'])]
    noKC = [ann['description'] for ann in annots if re.match(regex_noKC, ann['description'])]
    return len(KC), len(noKC)


def assert_points_position(window, window_filter, min_index, secondmax_index, start_of_event, max_index, end_of_event, window_length, sfreq):
    try:
        assert start_of_event >= 0, "Start of event found is delayed more than window length"
        assert start_of_event < secondmax_index, "Start of event found is bigger than second maximum of the window"
        assert secondmax_index > 0, "Second maximum found is delayed more than window length"
        assert secondmax_index <= min_index, "Second maximum of the window found is bigger than minimum of the window"
        assert min_index < max_index, "Minimum of the window found is bigger than maximum of the window"
        assert max_index <= end_of_event, "Maximum of the window found is bigger than end of the event"
    except AssertionError as msg:
        print(start_of_event, secondmax_index, min_index, max_index, end_of_event)
        t = np.linspace(0, window_length, int(window_length*sfreq))
        plt.plot(t, window)
        plt.plot(t, window_filter, 'gray', alpha=0.5)
        plt.plot(t[min_index], window[min_index], 'ro', label='minimo')
        plt.plot(t[secondmax_index], window[secondmax_index], 'bo', label='sec_max')
        plt.plot(t[start_of_event], window[start_of_event], 'go', label='start')
        plt.plot(t[max_index], window[max_index], 'yo', label='max')
        plt.plot(t[end_of_event], window[end_of_event], 'ko', label='end')
        plt.legend()
        plt.show(block=True)
        raise AssertionError(msg)
    return True


def detect_points_of_event(window, sfreq, signal, pos, window_length=3, min_interval=[1, 1.45], secmax_min_interval= 0.3, secmax_interval=[0.5, 1] , start_secmax_interval=0.5, start_min_interval=[0.2,1], max_interval=[1.55, 1.8], max_end_interval=[0.25,0.5], event_duration_threshold=[0.5, 2]):
    """
    """

    t = np.linspace(0, window_length, int(window_length*sfreq))
    window, new_pos = center_event_between_max_min(window, signal, sfreq, pos, new_window_length=window_length)
    sos = sp.signal.butter(2**5, [1, 8], 'bandpass', fs=sfreq, output='sos')
    window_filter = sp.signal.sosfiltfilt(sos, window)

    ### MINIMUM (in raw)
    idx_minimum = int(np.argmin(window[int(min_interval[0]*sfreq):int(min_interval[1]*sfreq)]))+int(min_interval[0]*sfreq)

    ### SEC MAX (in filtered)
    start_secmax_zone = idx_minimum - int(secmax_min_interval*sfreq)    
    window_to_look_for_secmax = window_filter[start_secmax_zone:idx_minimum]
    idx_secmax = int(np.argmax(window_to_look_for_secmax))+start_secmax_zone 
    if idx_secmax < int(secmax_interval[0]*sfreq) or idx_secmax > int(secmax_interval[1]*sfreq):
        idx_secmax = idx_minimum - int(secmax_min_interval*sfreq/2)

    ### START (in filtered)
    window1_to_look_for_start = window_filter[idx_secmax-int(start_secmax_interval*sfreq):idx_secmax]
    idx1_start = get_zc_nearest_to(window1_to_look_for_start, 'end', state='up') + idx_secmax-int(start_secmax_interval*sfreq)
    if idx1_start < idx_minimum-int(start_min_interval[1]*sfreq):
        window2_to_look_for_start = window_filter[idx_minimum-int(start_min_interval[1]*sfreq):idx_minimum]
        idx2_start = get_zc_nearest_to(window2_to_look_for_start, 'end', state='up') + idx_minimum-int(start_min_interval[1]*sfreq)
        if idx2_start > idx_secmax:
            idx_start = min(idx_minimum-int(start_min_interval[0]*sfreq), idx_secmax)
        else:
            idx_start = idx2_start
    else:
        idx_start = idx1_start

    ### MAXIMUM (in raw)
    idx_maximum = int(np.argmax(window[int(max_interval[0]*sfreq):int(max_interval[1]*sfreq)]))+int(max_interval[0]*sfreq) 

    ### END (in filtered)
    idx_end = get_zc_nearest_to(window_filter[idx_maximum:idx_maximum+int(max_end_interval[1]*sfreq)], 'start', state='down') + idx_maximum
    
    if (idx_end - idx_start) < int(event_duration_threshold[0]*sfreq):
        delay = (int(event_duration_threshold[0]*sfreq) - (idx_end - idx_start))//2
        idx_start -= delay
        idx_end = idx_start + int(event_duration_threshold[0]*sfreq)
    elif (idx_end - idx_start) > int(event_duration_threshold[1]*sfreq):
        idx1_end = idx_maximum + int(max_end_interval[0]*sfreq)
        idx2_end = int(event_duration_threshold[1]*sfreq) + idx_start
        idx_end = max(idx1_end, idx2_end)

    assert_points_position(window, window_filter, idx_minimum, idx_secmax, idx_start, idx_maximum, idx_end, window_length, sfreq)

    return window, idx_start, idx_minimum, idx_secmax, idx_maximum, idx_end, new_pos


def center_event_between_max_min(window, signal, sfreq, pos, new_window_length=3):
    """
    """
    max_index = int(np.argmax(window))
    min_index = int(np.argmin(window))
    assert max_index > min_index, "ERROR: Found maximum before minimum in KC candidate"
    center = pos + min_index + int((max_index-min_index)//2)
    new_window = signal[center-int(sfreq*(new_window_length/2)):center+int(sfreq*(new_window_length/2))]
    new_pos = center-int(sfreq*(new_window_length/2))
    return new_window, new_pos


def get_localized_and_detected(signal, sfreq: int, path_scoring: str, window_length: int =1, stages_allowed =[2.0], step = 0.1):
    """
    """
    stages_per_sample = set_sleep_stages_per_sample(signal, path_scoring, epoch_duration=30, sfreq=sfreq)    
    original_signal_length = len(signal)
    signal = signal[:len(stages_per_sample)]
    pos_candidate = []
    pos_predicted = []
    pos = 0
    t = np.linspace(0, window_length, int(window_length*sfreq))
    while pos < len(signal):
        if stages_per_sample[pos] in stages_allowed:
            window = signal[pos:pos+int(window_length*sfreq)]
            if check_if_meet_main_conditions(window, sfreq):
                try:
                    window, idx_start, idx_minimum, idx_secmax, idx_maximum, idx_end, new_pos = detect_points_of_event(window, sfreq, signal, pos, window_length=3) 
                except AssertionError: # Could not find the points of the event correctly
                    pos += int(window_length/2*sfreq)
                    continue
                pos_candidate.append([new_pos + idx_start, new_pos + idx_end])
                path = os.path.join(configuration.REPORTS_ROOT, 'models', 'SVM_C0.75_rbf_gammaScale_FirstLabelData.joblib')
                model = joblib.load(path)
                center = len(window)//2
                window_to_predict = window[center-int(sfreq):center+int(sfreq)]
                #window_to_predict = sp.signal.resample(window[idx_start:idx_end], int(2*sfreq))
                isKC = model.predict(window_to_predict.reshape(1, -1))
                if isKC == 1:
                    pos_predicted.append([new_pos + idx_start, new_pos + idx_end])
                pos = new_pos + idx_end
            else:
                pos += int(step*sfreq)
        else:
            pos += int(30*sfreq)
    
    loc = np.zeros(original_signal_length)
    pred = np.zeros(original_signal_length)
    for p in pos_candidate:
        loc[p[0]:p[1]] = [50e-6]*len(loc[p[0]:p[1]])
    for p in pos_predicted:
        pred[p[0]:p[1]] = [50e-6]*len(pred[p[0]:p[1]])
        
    return loc, pred