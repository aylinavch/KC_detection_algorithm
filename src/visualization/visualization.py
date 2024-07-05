import mne

def plot(raw: mne.io.Raw, title: str):
    """
    Re-structuration of raw object into another Raw object that is needed.
    Always the montage are like EOG - EEG - (grid) - EMG.
    
    """
    n_channels = len(raw.ch_names)
    order = [i for i in range(n_channels)]
    scal = dict(eeg=20e-5, eog=40e-5, emg=40e-5, misc=1e-3)

    raw.plot(show_options = True,
             title = title,
             start = 0,                        # initial time to show
             duration = 30,                    # time window (sec) to plot in a given time
             n_channels = n_channels, 
             scalings = scal,                  # scaling factor for traces.
             block = True,
             order = order
            )
    
    return raw