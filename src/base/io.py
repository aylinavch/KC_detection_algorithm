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
    """
    """
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


def clean_annotations_to_KC_noKC(new_annotations, orig_time):
    """
    """
    regex_KC = r"^KC(?:_\w+)?$"
    regex_noKC = r"^noKC(?:_\w+)?$"

    # Create a new list to store the modified annotations
    annotations_to_delete = []
    annotations_to_rename = []

    # Iterate over the annotations
    for i, ann in enumerate(new_annotations):
        # Check if the current annotation is "KC"
        if re.match(regex_KC, ann['description']):
            # Check for overlapping with "noKC" annotations
            for j, other_ann in enumerate(new_annotations):
                if re.match(regex_noKC, other_ann['description']):
                    # Check for overlap (noKC annotation starts during the other)
                    if ann['onset'] <= other_ann['onset'] < ann['onset'] + ann['duration']:
                        # Rename "noKC" to "KC"
                        annotations_to_rename.append(j)
                        annotations_to_delete.append(i)
                        break  # Stop searching once an overlap is found          
    try:
        new_KC_ann = new_annotations[annotations_to_rename].rename({'noKC': 'KC'})
        new_KC_ann_onset = [ann['onset'] for ann in new_KC_ann]
        new_KC_ann_duration = [ann['duration'] for ann in new_KC_ann]
        new_KC_ann_description = [ann['description'] for ann in new_KC_ann]
    except ValueError:
        print('\n\n WARNING: No KC annotations were found to rename to noKC as KC.\n\n')
        new_KC_ann = None

    new_annotations.delete(annotations_to_delete+annotations_to_rename)
    
    if new_KC_ann:
        new_annotations.append(new_KC_ann_onset, new_KC_ann_duration, new_KC_ann_description)
    
    # Create a new mne.Annotations object with the updated annotations
    new_annotations_cleaned = mne.Annotations(onset=[ann['onset'] for ann in new_annotations],
                                              duration=[ann['duration'] for ann in new_annotations],
                                              description=[ann['description'] for ann in new_annotations],
                                              orig_time=orig_time)

    return new_annotations_cleaned



def check_file_if_ready_to_save(raw_after_plot: mne.io.Raw, raw_before_plot: mne.io.Raw, annotations_path: str, mode: str, codename: str, subject: str):
    """
    """
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
            # raw_before_plot = get_only_KC_noKC_labels(raw_before_plot)
            if check_if_enough_candidates_were_labeled(raw_after_plot, mode, subject, codename):
                cleaned_annotations = clean_annotations_to_KC_noKC(raw_after_plot.annotations, raw_after_plot.annotations.orig_time)
                filename_output = os.path.join(configuration.ANNOTATIONS_ROOT, subject+f"_annotations_{codename}_{mode}.txt")
                print(f'\n\nSaving labels generated in {filename_output}...')
                cleaned_annotations.save(filename_output, overwrite=True)
                return True
            else:
                return False
        else:
            return False