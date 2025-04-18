# KC Detection Algorithm

K Complex detection algorithm based on Machine Learning Techniques for NREM-Stage 2 of sleep.

This algorithm is the continuation of the repository called "[KCdetector-S2NREM](https://github.com/aylinavch/KCdetector-S2NREM)" which was used for my Final Project to graduate in Bioengineering at Buenos Aires Institute of Technology.

> For any questions, contact _ayvazquez@itba.edu.ar_

## Set up 💻

You must need python 3.9 installed in your computer. Also you will need to create python environments  (`python-venv`).

Once you have both things, you must run in a terminal inside the directory of this project:

```
python3.9 -m venv .venv
```

If you have macOS or Linux:
```
source .venv/bin/activate
pip install requirements.txt
```

If you have Windows:

```
source .venv/Scripts/activate
pip install requirements.txt
```

Before running the program, you must create a configuration file called `configuration.py` with this variables:
- DB_ROOT: Path where database files are stored
- ANNOTATIONS_ROOT: Path where scoring and labeling files are stored
- REPORTS_ROOT: Path where outputs will be saved
- SUBJECTS: List with name if subjects to process
- EEG_CHANNEL: String with name of channel to use as reference for eeg)
- CUT_OFF_FREQUENCIES: Dictionary with frequencies (list of values) per channel (key) where pre-processing will be filter that band-width

> Check `configuration_example.py` for an example of this file

Once everything was installed, now you can run this program. 

> ⚠️ Everytime you will want to run this program again, you must activate the virtual environment. You can do it writing inside the console `source .venv/bin/activate` in Linux or MacOS, or running `source .venv/Scripts/activate` in Windows.


## Manual 📑

(not developed yet)


## Troubleshooting 🚫

Contact _ayvazquez@itba.edu.ar_
