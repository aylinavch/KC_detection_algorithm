import sys

def command_arguments_to_run_blind_labeling() -> bool:
    """
    This function verifies if user has called the program with the correct arguments to run the blind labeling.
    > COMMAND LINE: python labeling.py -blind
    """
    return (len(sys.argv) == 1) or (len(sys.argv) == 2 and sys.argv[-1]=='-blind') 

def command_arguments_to_run_semi_automatic_labeling() -> bool:
    """
    This function verifies if user has called the program with the correct arguments to run the semi-automatic labeling.
    > COMMAND LINE: python labeling.py -semiautomatic
    """
    return len(sys.argv) == 2 and sys.argv[-1]=='-semiautomatic'

def command_arguments_to_run_label_cleaning() -> bool:
    """
    This function verifies if user has called the program with the correct arguments to run the label cleaning.
    > COMMAND LINE: python labeling.py -cleaning
    """
    return len(sys.argv) == 2 and sys.argv[-1]=='-cleaning'