import configuration
import os
import mne
import re
from src.data.preprocess import set_annotations_labels, get_only_KC_noKC_labels
from src.base.localizator import count_KC_noKC


def check_if_there_is_old_annotation_file(annotations_path: str, raw: mne.io.Raw):
    """
    """
    if os.path.exists(annotations_path):
        ans = input('\nAn old annotation file was found.\n\nType "y" if you want to load its information to continue labeling: ')
        if ans.lower() == 'y':
             return set_annotations_labels(raw, annotations_path)
    else:
        print('\nNo old annotation file was found.\n')
    return raw


def check_if_same_num_of_KC_and_noKC(raw_only_KC_and_noKC: mne.io.Raw, mode: str, subject: str, codename: str):
    """
    """
    num_of_KC, num_of_noKC = count_KC_noKC(raw_only_KC_and_noKC)
    
    if num_of_KC != num_of_noKC:
        print('\nWARNING: The number of KC and noKC labels are not the same.\n',
               f'Num of KC = {num_of_KC}\n',
               f'Num of noKC = {num_of_noKC} \n\n')
        ans = input('Do you want to add missing labels in this file? (press "y" to add labels): ')
        if ans.lower() == 'y':
            return False
        else:
            return True
    return True


def check_if_there_is_new_labels(raw_after_plot, raw_before_plot, annotations_path, mode, subject, codename):
    
    if raw_after_plot.annotations != raw_before_plot.annotations:
        try:
            new_annotations_path = annotations_path.split('.')[0]+"_old.txt"
            os.rename(annotations_path, new_annotations_path)
        except FileNotFoundError:
            pass
        return True
    else:
        print('\nNo events were labeled in this file.\n')
        ans = input('Do you want to finish labeling? (type "n" to re-start this file labeling): ')
        if ans.lower() == 'n':
            return False
        else:
            return True


def check_if_enough_candidates_were_labeled(raw_after_plot: mne.io.Raw, mode: str, subject: str, codename: str):
    """
    """
    num_of_KC, num_of_noKC  = count_KC_noKC(raw_after_plot)
    
    if num_of_KC < int(0.1*num_of_noKC):
        print('\nWARNING: Number of candidates labeled as KC seems to be very few.\n',
               f'Num of KC = {num_of_KC}\n',
               f'Num of noKC = {num_of_noKC}\n\n')
        ans = input('Do you want to add more KC labels in this file? (press "y" to add labels): ')
        if ans.lower() == 'y':
            return False
        else:
            return True
    return True


def clean_CAND_annotations_to_KC_noKC(CAND_annotations, CAND_KC_annotations, orig_time):
    """
    """
    regex_cand = r"^noKC(?:_\w+)?$"
    regex_KC = r"^KC(?:_\w+)?$"
    
    KC_description = [ann['description'] for ann in CAND_KC_annotations if re.match(regex_KC, ann['description'])]
    KC_onset = [ann['onset'] for ann in CAND_KC_annotations if re.match(regex_KC, ann['description'])]
    KC_duration = [ann['duration'] for ann in CAND_KC_annotations if re.match(regex_KC, ann['description'])]
    KC_annotations = mne.Annotations(onset=KC_onset, 
                                     duration=KC_duration, 
                                     description=KC_description, 
                                     orig_time=orig_time)
    
    noKC_description = [ann['description'] for ann in CAND_KC_annotations if re.match(regex_cand, ann['description'])]
    noKC_onset = [ann['onset'] for ann in CAND_KC_annotations if re.match(regex_cand, ann['description'])]
    noKC_duration = [ann['duration'] for ann in CAND_KC_annotations if re.match(regex_cand, ann['description'])]
    noKC_annotations = mne.Annotations(onset=noKC_onset, 
                                     duration=noKC_duration, 
                                     description=noKC_description, 
                                     orig_time=orig_time)
    
    annot_to_KC = []

    for i, annot in enumerate(CAND_KC_annotations):
        if re.match(regex_cand, annot['description']):
            for KC_annot in KC_annotations:
                KC_start = KC_annot['onset']
                KC_end = KC_start + KC_annot['duration']
                if KC_start <= annot['onset'] <= KC_end:
                    annot_to_KC.append(i)
                    continue
    
    KC_annotations = CAND_KC_annotations[annot_to_KC]
    KC_annotations = mne.Annotations(onset=KC_annotations.onset, 
                                      duration=KC_annotations.duration, 
                                      description=['KC']*len(KC_annotations.onset), 
                                      orig_time=orig_time)
    
    
    CAND_annotations.delete(annot_to_KC)

    noKC_annotations = mne.Annotations(onset=CAND_annotations.onset, 
                                      duration=CAND_annotations.duration, 
                                      description=['noKC']*len(CAND_annotations.onset), 
                                      orig_time=orig_time)
    return noKC_annotations + KC_annotations


def check_file_if_ready_to_save(raw_after_plot: mne.io.Raw, raw_before_plot: mne.io.Raw, annotations_path: str, mode: str, codename: str, subject: str):
    
    if mode == 'blind':
        if check_if_there_is_new_labels(raw_after_plot, raw_before_plot, annotations_path, mode, subject, codename):
            raw_only_KC_and_noKC = get_only_KC_noKC_labels(raw_after_plot)
            if check_if_same_num_of_KC_and_noKC(raw_only_KC_and_noKC, mode, subject, codename):
                filename_output = os.path.join(configuration.ANNOTATIONS_ROOT, subject+f"_annotations_{codename}_{mode}.txt")
                print(f'\n\nSaving labels generated in {filename_output}...')
                raw_only_KC_and_noKC.annotations.save(filename_output, overwrite=True)
                return True
            else:
                return False
        else:
            return False
    elif mode == 'semiauto':
        if check_if_there_is_new_labels(raw_after_plot, raw_before_plot, annotations_path, mode, subject, codename):
            raw_after_plot = get_only_KC_noKC_labels(raw_after_plot)
            raw_before_plot = get_only_KC_noKC_labels(raw_before_plot)
            if check_if_enough_candidates_were_labeled(raw_after_plot, mode, subject, codename):
                cleaned_annotations = clean_CAND_annotations_to_KC_noKC(raw_before_plot.annotations, raw_after_plot.annotations, raw_after_plot.annotations.orig_time)
                filename_output = os.path.join(configuration.ANNOTATIONS_ROOT, subject+f"_annotations_{codename}_{mode}.txt")
                print(f'\n\nSaving labels generated in {filename_output}...')
                cleaned_annotations.save(filename_output, overwrite=True)
                return True
            else:
                return False
        else:
            return False

    
   