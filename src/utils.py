"""
Helper and utility functions for file handling, naming and MRS operations.
"""

import os
from pathlib import Path
import numpy as np
import fsl_mrs.utils.mrs_io as mrs_io
from fsl_mrs.utils.misc import calculateAxes


def get_directory_names(CSI_dir):
    path = os.path.normpath(CSI_dir)
    components = path.split(os.sep)
    return components[-3:-1]


def check_directory_exists(path):
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        print(f"Directory {path} created.")
    else:
        print(f"Directory {path} already exists.")


def load_basis_lip():
    basis = mrs_io.read_basis('PRESS_basis_WL_reduced')
    basis.add_peak(ppm=1.3, amp=10, name='Lip', gamma=2, sigma=2, conj=False)
    return basis


def gen_ppm_axis(CSI_data):
    ppmaxis = calculateAxes(
        CSI_data.spectralwidth,
        CSI_data.spectrometer_frequency[0],
        CSI_data.shape[3],
        0
    )['ppm']
    return ppmaxis + 4.65


def get_unique_filename(filepath):
    directory, basename = os.path.split(filepath)
    parts_filename = basename.split('.')
    filename = parts_filename[0]
    extension = '.'.join(parts_filename[1:])

    index = 0
    new_filepath = filepath
    while os.path.exists(new_filepath):
        index += 1
        if extension != '':
            new_filename = f"{filename}{index}.{extension}"
        else:
            new_filename = f"{filename}{index}"
        new_filepath = os.path.join(directory, new_filename)
    
    return new_filepath
