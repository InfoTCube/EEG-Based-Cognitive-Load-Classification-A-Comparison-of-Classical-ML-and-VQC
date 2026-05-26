import re
import numpy as np
import mne
from pathlib import Path


def load_raw_data(data_dir):
    """
    Load EEG recordings from EDF files as continuous signals.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing Subject##_[1|2].edf files.

    Returns
    -------
    recordings : list of dict, each with keys:
        'data'         ndarray (n_channels, n_samples) in µV
        'subject_id'   int
        'session_type' int  — 1 = rest, 2 = mental math
        'sfreq'        float
        'ch_names'     list[str]
    """
    edf_files = sorted(Path(data_dir).glob('Subject*.edf'))
    if not edf_files:
        raise ValueError(f"No EDF files found in {data_dir}")

    recordings = []
    for edf_path in edf_files:
        match = re.match(r'Subject(\d+)_(\d)\.edf', edf_path.name)
        if not match:
            continue

        subject_id   = int(match.group(1))
        session_type = int(match.group(2))

        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        raw.filter(1, 50, verbose=False)

        recordings.append({
            'data':         raw.get_data() * 1e6,  # V -> uV
            'subject_id':   subject_id,
            'session_type': session_type,
            'sfreq':        raw.info['sfreq'],
            'ch_names':     raw.ch_names,
        })

    return recordings
