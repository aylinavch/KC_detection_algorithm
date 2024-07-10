import datetime
import mne
import numpy as np
import os
import re
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colormaps as cm
from src.features.build_features_utils import get_center_maxi_mini, get_power, get_maxi_mini_slope, build_data_features

def get_KC_event(annotation, signal, sfreq, window=2, timelocked2='center'):
    """
    """
    if timelocked2 == 'center':
        start = int(annotation['onset']*sfreq)
        end = int(start + annotation['duration']*sfreq)
        KC_labeled = signal[start:end]

        center, maxi, mini = get_center_maxi_mini(KC_labeled)
        center_in_signal = start + center
        mini_to_center = abs(mini-center)
        maxi_to_center = abs(maxi-center)
        KC = signal[int(center_in_signal-window//2*sfreq):int(center_in_signal+window//2*sfreq)]
    elif timelocked2 == 'min':
        start = int(annotation['onset']*sfreq)
        end = int(start + annotation['duration']*sfreq)
        KC_labeled = signal[start:end]
        center, maxi, mini = get_center_maxi_mini(KC_labeled)
        center_in_signal = start + mini
        mini_to_center = 0
        maxi_to_center = abs(maxi-mini)
        KC = signal[int(center_in_signal-window//2*sfreq):int(center_in_signal+window//2*sfreq)]

    return KC, maxi_to_center, mini_to_center

def get_noKC_event(annotation, signal, sfreq, window=2):
    """
    """
    start = int(annotation['onset']*sfreq)
    end = int(start + annotation['duration']*sfreq)
    center = (start + end) // 2
    noKC = signal[center-int(window*sfreq/2):center+int(window*sfreq/2)]

    return noKC


def plot_events(raw: mne.io.Raw, channel_name: str, subject: str, reports_path: str):
    """
    """
    signal = raw.get_data(picks=channel_name)[0]
    all_annotations = raw.annotations
    regex_KC = r"^KC(?:_\w+)?$"
    regex_noKC = r"^noKC(?:_\w+)?$"

    KC_onset = [ann['onset'] for ann in all_annotations if re.match(regex_KC, ann['description'])]
    KC_duration = [ann['duration'] for ann in all_annotations if  re.match(regex_KC, ann['description'])]
    KC_description = [ann['description'] for ann in all_annotations if re.match(regex_KC, ann['description'])]
    KC_annotations = mne.Annotations(KC_onset, KC_duration, KC_description, orig_time=raw.info['meas_date'])
    KC_signal = []
    plt.figure(figsize=(10, 5))
    for KC_annot in KC_annotations:
        KC, maxi_to_center, mini_to_center = get_KC_event(KC_annot, signal, raw.info['sfreq'], window=2, timelocked2='min')
        # center = len(KC)//2
        # max_position = int(center - maxi_to_center)
        # min_position = int(center + mini_to_center)
        t = np.linspace(0, 2, len(KC))
        KC_signal.append(KC)
        plt.plot(t, KC*1e6, 'black', alpha=0.1)
        # plt.plot(t[max_position], KC[max_position], 'ro')
        # plt.plot(t[min_position], KC[min_position], 'bo')
    KC_mean = np.mean(KC_signal, axis=0)
    plt.plot(t, KC_mean*1e6, 'red', linewidth=2)
    plt.title(f'KC events in {subject}')
    # plt.show(block=True)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV)')
    plt.ylim(-150, 150)
    path_KC = os.path.join(reports_path, f'{subject}_KC.png')
    plt.savefig(path_KC)


    noKC_onset = [ann['onset'] for ann in all_annotations if re.match(regex_noKC, ann['description'])]
    noKC_duration = [ann['duration'] for ann in all_annotations if  re.match(regex_noKC, ann['description'])]
    noKC_description = [ann['description'] for ann in all_annotations if re.match(regex_noKC, ann['description'])]
    noKC_annotations = mne.Annotations(noKC_onset, noKC_duration, noKC_description, orig_time=raw.info['meas_date'])
    noKC_signal = []
    plt.figure(figsize=(10, 5))
    for noKC_annot in noKC_annotations:
        noKC = get_noKC_event(noKC_annot, signal, raw.info['sfreq'], window=2)
        t = np.linspace(0, 2, len(noKC))
        noKC_signal.append(noKC)
        plt.plot(t, noKC*1e6, 'black', alpha=0.1)
    noKC_mean = np.mean(noKC_signal, axis=0)
    plt.plot(t, noKC_mean*1e6, 'red', linewidth=2)
    plt.title(f'noKC events in {subject}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV)')
    plt.ylim(-150, 150)
    # plt.show(block=True)
    path_noKC = os.path.join(reports_path, f'{subject}_noKC.png')
    plt.savefig(path_noKC)

def get_events(raw: mne.io.Raw, channel_name: str, subject: str, reports_path: str, timelocked2='center'):
    signal = raw.get_data(picks=channel_name)[0]
    all_annotations = raw.annotations
    regex_KC = r"^KC(?:_\w+)?$"
    regex_noKC = r"^noKC(?:_\w+)?$"

    KC_onset = [ann['onset'] for ann in all_annotations if re.match(regex_KC, ann['description'])]
    KC_duration = [ann['duration'] for ann in all_annotations if  re.match(regex_KC, ann['description'])]
    KC_description = [ann['description'] for ann in all_annotations if re.match(regex_KC, ann['description'])]
    KC_annotations = mne.Annotations(KC_onset, KC_duration, KC_description, orig_time=raw.info['meas_date'])
    KC_signal = []
    for KC_annot in KC_annotations:
        KC, _, _ = get_KC_event(KC_annot, signal, raw.info['sfreq'], window=2, timelocked2=timelocked2)
        KC_signal.append(KC)
    path_KC = os.path.join(reports_path, f'{subject}_KC_timelocked2{timelocked2}.npy')
    np.save(path_KC, np.array(KC_signal))


    noKC_onset = [ann['onset'] for ann in all_annotations if re.match(regex_noKC, ann['description'])]
    noKC_duration = [ann['duration'] for ann in all_annotations if  re.match(regex_noKC, ann['description'])]
    noKC_description = [ann['description'] for ann in all_annotations if re.match(regex_noKC, ann['description'])]
    noKC_annotations = mne.Annotations(noKC_onset, noKC_duration, noKC_description, orig_time=raw.info['meas_date'])
    noKC_signal = []
    for noKC_annot in noKC_annotations:
        noKC = get_noKC_event(noKC_annot, signal, raw.info['sfreq'], window=2)
        noKC_signal.append(noKC)
    path_noKC = os.path.join(reports_path, f'{subject}_noKC.npy')
    np.save(path_noKC, np.array(noKC_signal))

def read_events(subject:str, reports_path: str):
    """
    """
    path_KC_timelocked2center = os.path.join(reports_path, f'{subject}_KC_timelocked2center.npy')
    path_KC_timelocked2min = os.path.join(reports_path, f'{subject}_KC_timelocked2min.npy')
    path_noKC = os.path.join(reports_path, f'{subject}_noKC.npy')
    
    KC_timelocked2center = np.load(path_KC_timelocked2center)
    KC_timelocked2min = np.load(path_KC_timelocked2min)
    noKC = np.load(path_noKC)

    return KC_timelocked2center, KC_timelocked2min, noKC

def save_mean_figures(reports_path, all_KCs_timelocked2center, all_KCs_timelocked2min, all_noKCs):
    """
    """
    t = np.linspace(0, 2, len(all_KCs_timelocked2center[0]))
        
    plt.figure(figsize=(16, 8))
    for kc in all_KCs_timelocked2center:
        plt.plot(t, kc*1e6, 'black', alpha=0.1)
    KC_timelocked2center_mean = np.mean(all_KCs_timelocked2center, axis=0)
    plt.plot(t, KC_timelocked2center_mean*1e6, 'red', linewidth=2, label='Mean')
    plt.title(f'KC events (N={len(all_KCs_timelocked2center)}) - Time locked to the middle of minimum and maximum absolute')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV)')
    plt.legend()
    plt.ylim(-200, 200)
    # plt.show(block=True)
    KC_timelocked2center_path = os.path.join(reports_path, f'KC_timelocked2center.png')
    plt.savefig(KC_timelocked2center_path)

    plt.figure(figsize=(16, 8))
    for kc in all_KCs_timelocked2min:
        plt.plot(t, kc*1e6, 'black', alpha=0.1)
    KC_timelocked2min_mean = np.mean(all_KCs_timelocked2min, axis=0)
    plt.plot(t, KC_timelocked2min_mean*1e6, 'red', linewidth=2, label='Mean')
    plt.title(f'KC events (N={len(all_KCs_timelocked2min)}) - Time locked to the minimun absolute')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV)')
    plt.ylim(-200, 200)
    plt.legend()
    # plt.show(block=True)
    KC_timelocked2min_path = os.path.join(reports_path, f'KC_timelocked2min.png')
    plt.savefig(KC_timelocked2min_path)

    plt.figure(figsize=(16, 8))
    for nokc in all_noKCs:
        plt.plot(t, nokc*1e6, 'black', alpha=0.1)
    noKC_mean = np.mean(all_noKCs, axis=0)
    plt.plot(t, noKC_mean*1e6, 'red', linewidth=2, label='Mean')
    plt.title(f'no-KC events (N={len(all_noKCs)})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV)')
    plt.ylim(-200, 200)
    plt.legend()
    # plt.show(block=True)
    noKC_path = os.path.join(reports_path, f'noKC.png')
    plt.savefig(noKC_path)

def plot_3d_figures(all_KCs_timelocked2center, all_KCs_timelocked2min, all_noKCs):
    """
    """
    t = np.linspace(0, 2, len(all_KCs_timelocked2center[0]))

    colormap = cm.get_cmap('turbo')

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i, evento in enumerate(all_KCs_timelocked2center):
        color = colormap(i)
        ax.plot(t, [i] * len(all_KCs_timelocked2center[0]), evento*1e6, color=color)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Evento')
    ax.set_zlabel('Amplitud (μV)')
    ax.set_title('KCs time locked to the middle of minimum and maximum absolute')
    plt.show(block=True)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i, evento in enumerate(all_KCs_timelocked2min):
        color = colormap(i)
        ax.plot(t, [i] * len(all_KCs_timelocked2min[0]), evento*1e6, color=color)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Evento')
    ax.set_zlabel('Amplitud (μV)')
    ax.set_title('KCs time locked to the minimun absolute')
    plt.show(block=True)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i, evento in enumerate(all_noKCs):
        color = colormap(i)
        ax.plot(t, [i] * len(all_noKCs[0]), evento*1e6, color=color)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Evento')
    ax.set_zlabel('Amplitud (μV)')
    ax.set_title('no-KCs')
    plt.show(block=True)

def save_characteristics_csv(reports_path, all_KCs_timelocked2center, all_KCs_timelocked2min, all_noKCs, sfreq):
    """
    """
    cols = ['power_SO', 'power_SO_relative', 
            'power_delta', 'power_delta_relative', 
            'power_sigma', 'power_sigma_relative', 
            'maxi', 'idx_maxi', 
            'mini', 'idx_mini', 
            'slope_positive', 
            'second_maxi', 'idx_second_maxi', 
            'slope_negative', 
            'num_of_zc_min_max', 
            'num_of_zc_max_min', 'kurtosis', 
            'skewness',
            'KC']
    
    df1 = pd.DataFrame(columns=cols)
    df2 = pd.DataFrame(columns=cols)
    
    for kc in all_KCs_timelocked2center:
        data = build_data_features(kc, sfreq)
        data.append('yes')
        df1.loc[len(df1)] = pd.Series(data, index=df1.columns)

    for kc in all_KCs_timelocked2min:
        data = build_data_features(kc, sfreq)
        data.append('yes')
        df2.loc[len(df2)] = pd.Series(data, index=df2.columns)

    for nokc in all_noKCs:
        data = build_data_features(nokc, sfreq)
        data.append('no')
        df1.loc[len(df1)] = pd.Series(data, index=df1.columns)
        df2.loc[len(df2)] = pd.Series(data, index=df2.columns)

    df1.to_csv(os.path.join(reports_path, 'characteristics_timelocked2center.csv'))
    df2.to_csv(os.path.join(reports_path, 'characteristics_timelocked2min.csv'))


