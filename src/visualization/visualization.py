import mne
import numpy as np
from scipy.signal import square

def pulse(N, sfreq):
    """
    Create artificial signal with a 0.5 sec pulse to do the grid on the interface
    """
    t = np.linspace(0, round(N/sfreq), N, endpoint=False) 
    signal_pulse = square(2 * np.pi * 1 * t)
    return signal_pulse


def add_grid_to_raw(raw: mne.io.Raw):
    """
    """
    new_data = raw.get_data().copy()
    new_data = np.insert(new_data, 2, pulse(raw.n_times, raw.info['sfreq']), axis=0)

    new_ch_names = raw.ch_names.copy()
    new_ch_names.insert(2, 'grid')
    new_ch_types = raw.get_channel_types()
    new_ch_types.insert(2, 'misc')

    new_info = mne.create_info(new_ch_names, sfreq=raw.info['sfreq'], ch_types=new_ch_types)
    new_info.set_meas_date(raw.info['meas_date'])
    new_raw = mne.io.RawArray(new_data, new_info)
    new_raw.set_annotations(raw.annotations)

    return new_raw

def plot(raw: mne.io.Raw, title: str):
    """
    Re-structuration of raw object into another Raw object that is needed.
    Always the montage are like EOG - EEG - (grid) - EMG.
    
    """
    raw_to_plot = add_grid_to_raw(raw.copy())
    n_channels = len(raw_to_plot.ch_names)
    order = [i for i in range(n_channels)]
    scal = dict(eeg=20e-5, eog=40e-5, emg=40e-5, misc=1e-3)

    raw_to_plot.plot(show_options = True,
             title = title,
             start = 0,                        # initial time to show
             duration = 30,                    # time window (sec) to plot in a given time
             n_channels = n_channels, 
             scalings = scal,                  # scaling factor for traces.
             block = True,
             order = order
            )
    
    return raw_to_plot