import re
import scipy as sp
import numpy as np
import mne.io
from src.utils.sleep_stages_utils import set_sleep_stages_per_sample
from src.utils.localizator_utils import get_zc_nearest_to
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

def assert_points_position(window, window_filter, min_index, secondmax_index, start_of_event, max_index, end_of_event, window_length, sfreq):
    try:
        assert start_of_event >= 0, "Start of event found is delayed more than window length"
        assert start_of_event < secondmax_index, "Start of event found is bigger than second maximum of the window"
        assert secondmax_index > 0, "Second maximum found is delayed more than window length"
        assert secondmax_index < min_index, "Second maximum of the window found is bigger than minimum of the window"
        assert min_index < max_index, "Minimum of the window found is bigger than maximum of the window"
        assert max_index < end_of_event, "Maximum of the window found is bigger than end of the event"
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

def detect_points_of_event(window, sfreq, signal, pos, window_length=1.5, min_interval=[1, 1.5], secmax_min_interval= [0.05,0.7], secmax_interval=[0.5, 1] , start_secmax_interval=0.15, max_end_interval=0.4):
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
    ### MINIMUM (in raw)
    idx_minimum = int(np.argmin(window))
    if idx_minimum < int(min_interval[0]*sfreq):
        delay_min = int(min_interval[0]*sfreq)-idx_minimum
        idx_minimum += delay_min
        pos -= delay_min
        window = signal[pos:pos+int(window_length*sfreq)]
    elif idx_minimum > int(min_interval[1]*sfreq):
        delay_min = int(min_interval[1]*sfreq)-idx_minimum
        idx_minimum += delay_min
        pos += delay_min
        window = signal[pos:pos+int(window_length*sfreq)]

    ### SEC MAX (in filtered)
    sos = sp.signal.butter(2**5, [1,8], 'bandpass', fs=sfreq, output='sos')
    window_filter = sp.signal.sosfiltfilt(sos, window)
    start_secmax_zone = int(secmax_interval[0]*sfreq) if int(secmax_interval[0]*sfreq) > 0 else 0
    end_secmax_zone = int(secmax_interval[1]*sfreq) if int(secmax_interval[1]*sfreq) < idx_minimum else idx_minimum - 1
    window_to_look_for_secmax = window_filter[start_secmax_zone:end_secmax_zone]
    idx_secmax = int(np.argmax(window_to_look_for_secmax))+start_secmax_zone 
    if (idx_secmax-idx_minimum)<int(secmax_min_interval[0]*sfreq) or (idx_secmax-idx_minimum)>int(secmax_min_interval[1]*sfreq):
        idx_secmax = start_secmax_zone + (start_secmax_zone-idx_minimum)//2

    ### START (in filtered)
    idx_start = get_zc_nearest_to(window_filter[:idx_secmax], 'end', state='up')

    ### MAXIMUM (in raw)
    idx_maximum = int(np.argmax(window[idx_minimum:]))+idx_minimum
    # if max_index > int((window_length-max_delayed)*sfreq):
    #     pos += int(max_delayed*sfreq)
    #     window = signal[pos:pos+int(window_length*sfreq)]
    #     window_filter = sp.signal.sosfiltfilt(sos, window)
    #     start_of_event = start_of_event - int(max_delayed*sfreq) if (start_of_event - int(max_delayed*sfreq)) > 0 else 0
    #     secondmax_index -= int(max_delayed*sfreq)
    #     min_index -= int(max_delayed*sfreq)
    #     if secondmax_index <= 0:
    #         secondmax_index = (min_index-start_of_event)//2
    #     max_index = int(np.argmax(window[min_index:]))+min_index
    

    ### Finding ending point (in window filtered)
    idx_end = get_zc_nearest_to(window_filter[idx_maximum:], 'start', state='down') + idx_maximum
    # if end_of_event == max_index:
    #     #print(f'End of event is the same as max_index ({end_of_event})')
    #     end_of_event = int(max_index + max_delayed*sfreq)
    #     #print('Se corrigió a ', end_of_event)

    #print(start_of_event, secondmax_index, min_index, max_index, end_of_event)
    assert_points_position(window, window_filter, idx_minimum, idx_secmax, idx_start, idx_maximum, idx_end, window_length, sfreq)

    # t = np.linspace(0, window_length, int(window_length*sfreq))
    # plt.plot(t, window)
    # plt.plot(t, window_filter, 'gray', alpha=0.5)
    # plt.plot(t[min_index], window[min_index], 'ro', label='minimo')
    # plt.plot(t[secondmax_index], window[secondmax_index], 'bo', label='sec_max')
    # plt.plot(t[start_of_event], window[start_of_event], 'go', label='start')
    # plt.plot(t[max_index], window[max_index], 'yo', label='max')
    # plt.plot(t[end_of_event], window[end_of_event], 'ko', label='end')
    # plt.legend()
    # plt.show(block=True)
    return window, idx_start, idx_minimum, idx_secmax, idx_maximum, idx_end, pos

def center_event_between_max_min(window, signal, sfreq, pos, window_length):
    """
    """
    max_index = int(np.argmax(window))
    min_index = int(np.argmin(window))
    assert max_index > min_index, "ERROR: Found maximum before minimum in KC candidate"
    center = pos + min_index + int((max_index-min_index)//2)
    new_window = signal[center-int(sfreq*(window_length/2)):center+int(sfreq*(window_length/2))]
    new_pos = center-int(sfreq*(window_length/2))
    return new_window, new_pos

def get_candidates(signal, sfreq: int, path_scoring: str, window_length: int =1.5, stages_allowed =[2.0], step = 0.1):
    """
    """
    stages_per_sample = set_sleep_stages_per_sample(signal, path_scoring, epoch_duration=30, sfreq=sfreq)    
    original_signal_length = len(signal)
    signal = signal[:len(stages_per_sample)]
    pos_candidate = []
    pos = 0
    t = np.linspace(0, window_length, int(window_length*sfreq))
    while pos < original_signal_length-int(window_length*sfreq):
        if stages_per_sample[pos] in stages_allowed:
            window = signal[pos:pos+int(window_length*sfreq)]
            if check_if_KC_candidate(window, sfreq):
                window, pos = center_event_between_max_min(window, signal, sfreq, pos, window_length)
                try:
                    window, start_of_event, min_index, secondmax_index, max_index, end_of_event, pos = detect_points_of_event(window, sfreq, signal, pos, window_length) 
                except AssertionError: # Could not find the points of the event correctly
                    pos += int(step*sfreq)
                    continue
                if ((end_of_event - start_of_event) < int(0.4*sfreq)) or ((end_of_event - start_of_event) > int(2.1*sfreq)): # Event is too short or too long
                    pos += int(step*sfreq)
                    continue
                sos = sp.signal.butter(2**3, [1,8], 'bandpass', fs=sfreq, output='sos')
                window_filter = sp.signal.sosfiltfilt(sos, window)
                plt.plot(t, window)
                plt.plot(t, window_filter, 'gray', alpha=0.5)
                plt.plot(t[min_index], window[min_index], 'ro', label='min')
                plt.plot(t[secondmax_index], window[secondmax_index], 'bo', label='sec_max')
                plt.plot(t[start_of_event], window[start_of_event], 'go', label='start')
                plt.plot(t[max_index], window[max_index], 'yo', label='max')
                plt.plot(t[end_of_event], window[end_of_event], 'ko', label='end')
                plt.legend()
                plt.show(block=True)

                pos_start = pos + start_of_event
                assert end_of_event > 0
                pos_end = pos + end_of_event
                pos_candidate.append([pos_start, pos_end])
                pos = pos_end + 1
            else:
                pos += int(step*sfreq)
        else:
            pos += int(30*sfreq)
    
    flags = np.zeros(original_signal_length)
    for p in pos_candidate:
        flags[p[0]:p[1]] = [50e-6]*len(flags[p[0]:p[1]])

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