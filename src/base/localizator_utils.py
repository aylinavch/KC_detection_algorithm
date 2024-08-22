import numpy as np
import scipy as sp


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


def get_num_of_zc(signal, state='up'):
    all_zc = []
    for pos in range(len(signal)-1):
        if state == 'up':
            condition = (signal[pos] < 0 and signal[pos + 1] > 0)
        else: 
            condition = (signal[pos] > 0 and signal[pos + 1] < 0)
        if condition : #zero-crossing upstate was found
            all_zc.append(pos)
    
    return len(all_zc)


def detect_points_of_KC(zoi, start_zoi, onset_of_label, duration_of_label, sfreq):
    """
    """
    offset = onset_of_label - start_zoi
    assert offset > 0 
    event = zoi[offset:offset+int(duration_of_label*sfreq)]
    try:
        sos = sp.signal.butter(2**8, [1,8], 'bandpass', fs=sfreq, output='sos')
        zoi_filtered = sp.signal.sosfiltfilt(sos, zoi)
    except ValueError:
        zoi_filtered = zoi
    idx_start_label_in_zoi = offset
    event_filtered = zoi_filtered[offset:offset+int(duration_of_label*sfreq)]

    # Minimum
    idx_minimum_in_label = int(np.argmin(event))
    idx_minimum_in_zoi = idx_minimum_in_label + offset

    # Second maximum
    idx_sec_maximum_in_label = int(np.argmax(event[:idx_minimum_in_label]))
    idx_sec_maximum_in_zoi = idx_sec_maximum_in_label + offset
    
    # "Start of event" (ZC NEAREST TO SECOND MAXIMUM)
    idx_start_in_label = get_zc_nearest_to(event_filtered[:idx_sec_maximum_in_label], to='end', state='up')
    idx_start_in_zoi = idx_start_in_label + offset
    num_of_zc_until_minimum = get_num_of_zc(event_filtered[:idx_minimum_in_label], state='up')
    num_of_zc_until_sec_maximum = get_num_of_zc(event_filtered[:idx_minimum_in_label], state='up')

    # Maximum
    idx_maximum_in_label = int(np.argmax(event))
    idx_maximum_in_zoi = idx_maximum_in_label + offset
    if idx_maximum_in_label < idx_minimum_in_label:
        print("ERROR: Maximum is not after minimum. Maybe this label was wrong created.")

    # End of event
    idx_end_in_label = get_zc_nearest_to(event_filtered[idx_maximum_in_label:], to='start', state='down') + idx_maximum_in_label
    idx_end_in_zoi = idx_end_in_label + offset
    idx_end_label_in_zoi = idx_start_label_in_zoi + int(duration_of_label*sfreq)
    num_of_zc_after_minimum = get_num_of_zc(event_filtered[:idx_minimum_in_label], state='down')
    num_of_zc_after_maximum = get_num_of_zc(event_filtered[:idx_sec_maximum_in_label], state='down')

    return idx_start_label_in_zoi, idx_start_in_zoi, idx_sec_maximum_in_zoi, idx_minimum_in_zoi, idx_maximum_in_zoi, idx_end_label_in_zoi, idx_end_in_zoi, num_of_zc_until_minimum, num_of_zc_until_sec_maximum, num_of_zc_after_minimum, num_of_zc_after_maximum
    

    