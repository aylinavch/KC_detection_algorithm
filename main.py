import os
import configuration
from src.data.load_data import load_file

for subject in configuration.SUBJECTS:

    file_path = os.path.join(configuration.DB_ROOT, subject+'.vhdr')

    raw = load_file(file_path)
    print('... Finish file reading ...')