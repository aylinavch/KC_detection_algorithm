import configuration
import os
import sys

from src.utils.command_arguments_utils import command_arguments_to_run_blind_labeling, command_arguments_to_run_semi_automatic_labeling, command_arguments_to_run_automatic_labeling
from src.utils.labeling_utils import run_blind_labeling, run_semi_automatic_labeling, run_automatic_labeling

if __name__ == '__main__':
    """
    Two modes:
    1) -labeling (default): Plot signal and clean/label K-Complexes (with localizator mode on)
    2) -cleaning: Clean K-Complexes (duration between 0.5s and 2s; no duplicated)
    """

    labeler_codename = input('\n\nWrite your codename: ')

    for subject in configuration.SUBJECTS:
        print(f'\n\n\n-------------------------------------{subject}-------------------------------------\n')

        if command_arguments_to_run_blind_labeling():
            run_blind_labeling(subject, labeler_codename)

        elif command_arguments_to_run_semi_automatic_labeling():
            run_semi_automatic_labeling(subject, labeler_codename)

        elif command_arguments_to_run_automatic_labeling():
            run_automatic_labeling(subject, labeler_codename)

        else:
            print('\nInvalid command arguments')
            