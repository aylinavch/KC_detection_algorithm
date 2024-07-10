import numpy as np
from scipy import integrate
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from matplotlib import pyplot as plt

def get_center_maxi_mini(KC):
    """
    """
    maxi = max(KC)
    maxi_pos = np.where(KC == maxi)[0][0]
    mini = min(KC)
    mini_pos = np.where(KC == mini)[0][0]
    center = (maxi_pos + mini_pos) // 2
    return center, maxi_pos, mini_pos

def get_power(signal, rythm, sfreq):
    """
    """
    if rythm == 'SO':
        cut_freqs = [0.5, 1]
    elif rythm == 'delta':
        cut_freqs = [1, 4]
    elif rythm == 'theta':
        cut_freqs = [4, 8]
    elif rythm == 'alpha':
        cut_freqs = [8, 13]
    elif rythm == 'sigma':
        cut_freqs = [9, 15]
    elif rythm == 'beta':
        cut_freqs = [15, 30]
    elif rythm == 'gamma':
        cut_freqs = [30, 60]

    nperseg=int(sfreq*2)
    freq_total, psd_total = welch(signal*1e6, fs=sfreq, nperseg=nperseg, noverlap=nperseg/2, nfft=nperseg*5)

    # plt.figure(figsize=(16, 8))
    # plt.plot(freq_total, psd_total)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude (μV²/Hz)')
    # plt.xlim([0,20])
    # plt.show(block=True)

    f_start, f_end = cut_freqs
    psd = psd_total[np.where((freq_total>=f_start) & (freq_total<=f_end))]
    freq = freq_total[np.where((freq_total>=f_start) & (freq_total<=f_end))]
    
    auc_total = integrate.simpson(y=psd_total, x=freq_total)
    auc_rythm = integrate.simpson(y=psd, x=freq)
    auc_rythm_relative = auc_rythm / auc_total

    return auc_rythm, auc_rythm_relative

def get_maxi_mini_slope(event, sfreq):
    """
    """
    mid = len(event)//2
    zone_of_interest_maxi = event[mid:mid+int(sfreq*0.25)]
    zone_of_interest_mini = event[mid-int(sfreq*0.25):mid]
    maxi = max(zone_of_interest_maxi)
    mini = min(zone_of_interest_mini)
    idx_maxi = int(np.where(zone_of_interest_maxi == maxi)[0][0]) + mid
    idx_mini = int(np.where(zone_of_interest_mini == mini)[0][0]) + mid - int(sfreq*0.25)
    slope_positive = ((maxi - mini) / (idx_maxi - idx_mini))*1e6

    zone_of_interest_second_maxi = event[idx_mini-int(0.25*sfreq):idx_mini]
    second_maxi = max(zone_of_interest_second_maxi)
    idx_second_maxi = int(np.where(zone_of_interest_second_maxi == second_maxi)[0][0]) + idx_mini - int(sfreq*0.25)
    slope_negative = ((mini - second_maxi) / (idx_mini - idx_second_maxi))*1e6

    return [maxi*1e6, idx_maxi], [mini*1e6, idx_mini], slope_positive, [second_maxi*1e6, idx_second_maxi], slope_negative

def build_data_features(kc, sfreq):
    """
    """
    power_SO, power_SO_relative = get_power(kc, rythm='SO', sfreq=sfreq)
    power_delta, power_delta_relative = get_power(kc, rythm='delta', sfreq=sfreq)
    power_sigma, power_sigma_relative = get_power(kc, rythm='sigma', sfreq=sfreq)
    [maxi, idx_maxi], [mini, idx_mini], slope_positive, [second_maxi, idx_second_maxi], slope_negative = get_maxi_mini_slope(kc, sfreq)
    num_of_zc_min_max = len(np.where(np.diff(np.sign(kc[idx_mini:idx_maxi])))[0])
    num_of_zc_max_min = len(np.where(np.diff(np.sign(kc[idx_second_maxi:idx_mini])))[0])
    kurt = kurtosis(kc)
    skewness = skew(kc)
    data = [power_SO, power_SO_relative, 
            power_delta, power_delta_relative, 
            power_sigma, power_sigma_relative, 
            maxi, idx_maxi, 
            mini, idx_mini, 
            slope_positive, 
            second_maxi, idx_second_maxi, 
            slope_negative, 
            num_of_zc_min_max, 
            num_of_zc_max_min,
            kurt, skewness]
    data = [round(d, 3) for d in data]
    # plt.figure(figsize=(16, 8))
    # t = np.arange(len(kc))
    # plt.plot(t, kc)
    # plt.plot(t[idx_maxi], kc[idx_maxi], 'ro')
    # plt.plot(t[idx_mini], kc[idx_mini], 'bo')
    # plt.plot(t[idx_second_maxi], kc[idx_second_maxi], 'go')
    # plt.show(block=True)

    return data