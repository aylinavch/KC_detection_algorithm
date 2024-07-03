import mne.io

def bandpass_filter(raw: mne.io.Raw, selected_channels: list, cut_off_frequencies: list):
    """
    Bandpass filter applied to specific channels indicating which cut off frequencies.

    Parameters
    ----------
    raw : raw.io.Raw
        Raw object from MNE containing the data

    selected_channels: list
        List containing the name of channels to bandpass filter

    cut_off_frequencies: list
        List with two elements, the first one is the cut off frequency of the highpass filter
        The second element is the cut off frequency of the lowpass filter
    
    Returns
    ----------
    raw_filtered : raw.io.Raw
        Raw object from MNE containing the data filtered
    """
    assert len(cut_off_frequencies) == 2, f"You must specify two elements in the list, {len(cut_off_frequencies)} were given"

    raw_filtered = raw.copy().filter(
        l_freq=cut_off_frequencies[0], h_freq=cut_off_frequencies[1],
        picks=selected_channels)

    return raw_filtered