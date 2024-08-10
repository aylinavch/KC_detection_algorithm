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

def get_zc_nearest_to(signal, to='end', state='up'):
    all_zc = []
    for pos in range(len(signal)-1):
        if state == 'up':
            condition = (signal[pos] < 0 and signal[pos + 1] > 0)
        else: 
            condition = (signal[pos] > 0 and signal[pos + 1] < 0)
        if condition : #zero-crossing upstate was found
            all_zc.append(pos)
    
    if len(all_zc) > 0:
        if to == 'end':
            return all_zc[-1]
        else:
            return all_zc[0]
    else:
        if to == 'end':
            return 0
        else:
            return len(signal)-1

def detect_points_of_event(window, sfreq, signal, pos, window_length, min_delayed=0.2, secmax_delayed=0.2, max_delayed=0.1):
    """
    start_of_event, , max_index, end_of_event
    
    min_index: Position of the minimum of the window 
        If position is after 0.25*sfreq just returns the minimum in the window.
        Otherwise the window is delayed 0.25 seconds from that minimum and this new position is returned taking into account this delayed.
    
    secondmax_index: Position of the second maximum of the window
        This maximum is looked for in the window filtered 0.2 seconds before the minimum.
    
    Start: First zero crossing or 0.2 seconds before the second max of the window or 0
    Return the zero-crossing point before the second maximum of the window.
    If not found, return the position 0.5 seconds before the second maximum.
    """
    assert len(window) > 0

    ### Finding minimum (in window)
    if int(np.argmin(window)) > int(min_delayed*sfreq):
        min_index = int(np.argmin(window))
    else: # delay window min_delayed seconds from minimum
        min_index = int(np.argmin(window))
        pos = min_index-int(min_delayed*sfreq)
        min_index += int(min_delayed*sfreq)
        window = signal[pos:pos+int(window_length*sfreq)]

    ### Finding second maximum (in window filtered)   
    sos = sp.signal.butter(2**3, [1,8], 'bandpass', fs=sfreq, output='sos')
    window_filter = sp.signal.sosfiltfilt(sos, window)

    start_event_until_min = min_index-int(min_delayed*sfreq)
    event_until_min = window_filter[start_event_until_min:min_index]
    secondmax_index = int(np.argmax(event_until_min))+start_event_until_min
    if secondmax_index < int(secmax_delayed*sfreq):
        secondmax_index = int(secmax_delayed/2*sfreq)
    # if secondmax_index == min_index:
    # int(np.argmax(event_until_min))+min_index-int(0.2*sfreq) if min_index-int(0.2*sfreq) > 0 else int(np.argmax(event_until_min))

    ### Finding starting point (in window filtered)
    start_of_event = get_zc_nearest_to(window_filter[:secondmax_index], 'end', state='up')
    ### Finding maximum (in window)
    max_index = int(np.argmax(window[min_index:]))+min_index
    if max_index > int((window_length-max_delayed)*sfreq):
        print('ENTRO')
        pos -= int(max_delayed*sfreq)
        min_index -= int(max_delayed*sfreq)
        secondmax_index -= int(max_delayed*sfreq)
        start_of_event -= int(max_delayed*sfreq)
        window = signal[pos:pos+int(window_length*sfreq)]
        window_filter = sp.signal.sosfiltfilt(sos, window)
        max_index = int(np.argmax(window[min_index:]))+min_index

    ### Finding ending point (in window filtered)
    end_of_event = get_zc_nearest_to(window_filter[max_index:], 'start', state='down') + max_index
        
    # plt.plot(t, window)
    # plt.plot(t[min_index], window[min_index], 'ro', label='minimo')
    # plt.plot(t[secondmax_index], window[secondmax_index], 'bo', label='sec_max')
    # plt.plot(t[start_of_event], window[start_of_event], 'go', label='start')
    # plt.plot(t[max_index], window[max_index], 'yo', label='max')
    # plt.plot(t[end_of_event], window[end_of_event], 'ko', label='end')
    # plt.legend()
    # plt.show(block=True)
    # position = pos if pos < int(secondmax_index - 0.2*sfreq) else int(secondmax_index - 0.2*sfreq)
    # position = position if position < min_index else 0
    #         # return position, min_index, secondmax_index
    # position = int(secondmax_index - 0.2*sfreq) if int(secondmax_index - 0.2*sfreq) < min_index else 0
    # return position, min_index, secondmax_index
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
    return window, start_of_event, min_index, secondmax_index, max_index, end_of_event, pos

# def detect_ending_point_of_event(window: np.ndarray, sfreq: int):
#     """
#     Return the zero-crossing point before the second maximum of the window.
#     If not found, return the position 0.5 seconds before the second maximum.
#     """
#     assert len(window) > 0 

#     sos = sp.signal.butter(2**3, [0.16,8], 'bandpass', fs=sfreq, output='sos')
#     window_filter = sp.signal.sosfiltfilt(sos, window)
#     min_index = int(np.argmin(window))
#     start_event = min_index-int(0.2*sfreq) if min_index-int(0.2*sfreq) > 0 else 0
#     event_until_min = window[start_event:min_index]
#     secondmax_index = min_index - int(0.2*sfreq) + int(np.argmax(event_until_min))

#     for pos in range(secondmax_index, 0,-1):
#         if (window_filter[pos] > 0 and window_filter[pos - 1] < 0):

#             return pos if pos < int(secondmax_index - 0.2*sfreq) else int(secondmax_index - 0.2*sfreq)
#     return int(secondmax_index - 0.2*sfreq)



def get_candidates_no_upsampling(signal, sfreq: int, path_scoring: str, window_length: int =1, stages_allowed =[2.0], step = 0.1, output_length=2):
    """
    """
    stages_per_sample = set_sleep_stages_per_sample(signal, path_scoring, epoch_duration=30, sfreq=sfreq)    
    original_signal_length = len(signal)
    signal = signal[:len(stages_per_sample)]
    pos_candidate = []
    pos = 0
    t = np.linspace(0, output_length, int(output_length*sfreq))
    while pos < len(signal):
        if stages_per_sample[pos] in stages_allowed:
            window = signal[pos:pos+int(window_length*sfreq)]
            if check_if_KC_candidate(window, sfreq):
                max_index = int(np.argmax(window))
                min_index = int(np.argmin(window))
                center = pos + min_index + int((max_index-min_index)//2)
                window = signal[center-int(sfreq*(window_length/2)):center+int(sfreq*(window_length/2))]
                pos = center-int(sfreq*(window_length/2))
                window, start_of_event, min_index, secondmax_index, max_index, end_of_event, pos = detect_points_of_event(window, sfreq, signal, pos, window_length) 
                
                if (end_of_event - start_of_event) < int(0.4*sfreq):
                    pos += end_of_event
                    continue

                center = pos + min_index + int((max_index-min_index)//2)
                new_window = signal[center-int(sfreq*(output_length/2)):center+int(sfreq*(output_length/2))]
                sos = sp.signal.butter(2**3, [1,8], 'bandpass', fs=sfreq, output='sos')
                new_window_filter = sp.signal.sosfiltfilt(sos, new_window)
                dif = pos  - center + int(sfreq*(output_length/2))
                plt.plot(t, new_window)
                plt.plot(t, new_window_filter, 'gray', alpha=0.5)
                plt.plot(t[min_index+dif], new_window[min_index+dif], 'ro', label='minimo')
                plt.plot(t[secondmax_index+dif], new_window[secondmax_index+dif], 'bo', label='sec_max')
                plt.plot(t[start_of_event+dif], new_window[start_of_event+dif], 'go', label='start')
                plt.plot(t[max_index+dif], new_window[max_index+dif], 'yo', label='max')
                plt.plot(t[end_of_event+dif], new_window[end_of_event+dif], 'ko', label='end')
                plt.legend()
                plt.show(block=True)
                pos_cand = pos + start_of_event
                pos_candidate.append(pos_cand)
                pos += end_of_event  
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