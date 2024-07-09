import re
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

    if (pos_mini<pos_maxi) and (p2p_amplitude>75) and (dur_mini_maxi<2):
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
                mini = min(window)
                pos_mini = pos+np.where(window==mini)[0][0]
                pos_candidate.append(pos_mini)
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