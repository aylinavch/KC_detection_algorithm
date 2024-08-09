import re
import scipy as sp
import numpy as np
import mne.io
from src.utils.sleep_stages_utils import set_sleep_stages_per_sample
from matplotlib import pyplot as plt

def count_zero_crossings(arr: np.ndarray) -> int:
    num_zc = 0
    for i in range(1, len(arr)):
        if (arr[i-1] > 0 and arr[i] < 0) or (arr[i-1] < 0 and arr[i] > 0):
            num_zc += 1
    return num_zc


def check_if_KC_candidate(window, sfreq: int):
    """
    """
    maxi = max(window)
    mini = min(window)
    pos_maxi = np.where(window==maxi)[0][0] 
    pos_mini = np.where(window==mini)[0][0]

    p2p_amplitude = (maxi - mini)*1e6
    dur_mini_maxi = (pos_maxi - pos_mini)/sfreq #in seconds

    if (pos_mini<pos_maxi) and (p2p_amplitude>75) and (dur_mini_maxi<0.5):
        return True
    else:
        return False
    # print((pos_mini<pos_maxi), (p2p_amplitude>75), (dur_mini_maxi<2))
    # return maxi, mini, pos_maxi, pos_mini, p2p_amplitude, dur_mini_maxi


def get_flags(signal, sfreq: int, path_scoring: str, window_length: int =1, stages_allowed =[2.0], step = 0.1):
    """
    """
    stages_per_sample = set_sleep_stages_per_sample(signal, path_scoring, epoch_duration=30, sfreq=sfreq)    
    original_signal_length = len(signal)
    signal = signal[:len(stages_per_sample)]
    pos_candidate = []
    pos = 0
    while pos < len(signal):
        if stages_per_sample[pos] in stages_allowed:
            window = signal[pos:pos+int(window_length*sfreq)]
            if check_if_KC_candidate(window, sfreq):
                pos_cand = pos+len(window)//2
                pos_candidate.append(pos_cand)
            # if pos > int(1139.465496*sfreq) and pos < int((1139.465496+0.5279472140764483)*sfreq):
            #     print(f'pos: {pos}')
            #     print(pos+int(step*sfreq))
            #     print(pos_mini)
        pos += int(step*sfreq)

    flags = np.zeros(original_signal_length)
    for p in pos_candidate:
        flags[p:p+int(step*sfreq)] = 50e-6
        
    return flags

        # window = signal[int(1600.904352*sfreq):int((1600.904352+0.5455454545453904)*sfreq)]
        # maxi, mini, pos_maxi, pos_mini, p2p_amplitude, dur_mini_maxi = check_if_KC_candidate(window, sfreq)
        # print(f'maxi: {maxi}, mini: {mini}, pos_maxi: {pos_maxi}, pos_mini: {pos_mini}, p2p_amplitude: {p2p_amplitude}, dur_mini_maxi: {dur_mini_maxi}')
        # t = np.linspace(1600.904352, 1600.904352+0.5455454545453904, len(window))

        # plt.plot(t, window)
        # plt.plot(t[pos_maxi], maxi, 'ro')
        # plt.plot(t[pos_mini], mini, 'bo')

def count_KC_noKC(raw: mne.io.Raw):
    """
    """
    annots = raw.annotations
    regex_KC = r"^KC(?:_\w+)?$"
    regex_noKC = r"^noKC(?:_\w+)?$"
    KC = [ann['description'] for ann in annots if re.match(regex_KC, ann['description'])]
    noKC = [ann['description'] for ann in annots if re.match(regex_noKC, ann['description'])]
    print('Cantidad de KC:', len(KC) )
    print('Cantidad de no KC:', len(noKC) )

def detect_starting_point_of_event(window: np.ndarray, sfreq: int):
    """
    Return the zero-crossing point before the second maximum of the window.
    If not found, return the position 0.5 seconds before the second maximum.
    """
    assert len(window) > 0 
    #t = np.linspace(0, len(window)/sfreq, len(window))
    sos = sp.signal.butter(2**3, [1,8], 'bandpass', fs=sfreq, output='sos')
    window_filter = sp.signal.sosfiltfilt(sos, window)
    min_index = int(np.argmin(window_filter))
    start_event = min_index-int(0.2*sfreq) if min_index-int(0.2*sfreq) > 0 else 0
    event_until_min = window_filter[start_event:min_index]
    secondmax_index = min_index - int(0.2*sfreq) + int(np.argmax(event_until_min))

    for pos in range(secondmax_index, 0,-1):
        if (window_filter[pos] > 0 and window_filter[pos - 1] < 0):
            position = pos if pos < int(secondmax_index - 0.2*sfreq) else int(secondmax_index - 0.2*sfreq)
            return position, min_index, secondmax_index
    return int(secondmax_index - 0.2*sfreq), min_index, secondmax_index

def detect_ending_point_of_event(window: np.ndarray, sfreq: int):
    """
    Return the zero-crossing point before the second maximum of the window.
    If not found, return the position 0.5 seconds before the second maximum.
    """
    assert len(window) > 0 

    sos = sp.signal.butter(2**3, [0.16,8], 'bandpass', fs=sfreq, output='sos')
    window_filter = sp.signal.sosfiltfilt(sos, window)
    min_index = int(np.argmin(window))
    start_event = min_index-int(0.2*sfreq) if min_index-int(0.2*sfreq) > 0 else 0
    event_until_min = window[start_event:min_index]
    secondmax_index = min_index - int(0.2*sfreq) + int(np.argmax(event_until_min))

    for pos in range(secondmax_index, 0,-1):
        if (window_filter[pos] > 0 and window_filter[pos - 1] < 0):

            return pos if pos < int(secondmax_index - 0.2*sfreq) else int(secondmax_index - 0.2*sfreq)
    return int(secondmax_index - 0.2*sfreq)

def get_candidates(signal, sfreq: int, path_scoring: str, window_length: int =1, stages_allowed =[2.0], step = 0.1):
    """
    """
    stages_per_sample = set_sleep_stages_per_sample(signal, path_scoring, epoch_duration=30, sfreq=sfreq)    
    original_signal_length = len(signal)
    signal = signal[:len(stages_per_sample)]
    pos_candidate = []
    pos = 0
    t = np.linspace(0, window_length, int(window_length*sfreq))
    while pos < len(signal):
        if stages_per_sample[pos] in stages_allowed:
            window = signal[pos:pos+int(window_length*sfreq)]
            if check_if_KC_candidate(window, sfreq):
                start_of_event, min_index, secondmax_index = detect_starting_point_of_event(window, sfreq)
                plt.plot(t, window)
                plt.plot(t[start_of_event], window[start_of_event], 'ro', label='inicio')
                plt.plot(t[min_index], window[min_index], 'bo', label='minimo')
                plt.plot(t[secondmax_index], window[secondmax_index], 'go', label='sec max')
                plt.legend()
                plt.show(block=True)
                pos_cand = pos + start_of_event
                pos_candidate.append(pos_cand)
                pos = int(pos_cand + 0.5*sfreq)    
            else:
                pos += int(step*sfreq)
        else:
            pos += int(30*sfreq)
    
    flags = np.zeros(original_signal_length)
    for p in pos_candidate:
        flags[p] = 50e-6

    # candidates = np.zeros(original_signal_length)
    # while pos < len(flags)-2*sfreq:
    #     if flags[pos] == 1:
    #         if 1 not in flags[pos:pos+2*sfreq]:
    #             candidates[pos:pos+2*sfreq] = 50e-6
    #         else:
    #             cant_of_flags = sum(flags[pos:pos+2*sfreq])
    #             if cant_of_flags:
    #             candidate = min(signal[pos:pos+2*sfreq])

        
    return flags