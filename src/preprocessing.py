import numpy as np
from sklearn.model_selection import train_test_split


def epoch_recordings(recordings, epoch_duration=2.0, overlap=0.5):
    """
    Slice continuous recordings into fixed-length overlapping epochs.

    Parameters
    ----------
    recordings : list of dict
        Output of data_loader.load_raw_data.
    epoch_duration : float
        Epoch length in seconds.
    overlap : float
        Fractional overlap between consecutive epochs (0.5 = 50%).

    Returns
    -------
    X : ndarray (n_epochs, n_channels, n_samples)
    labels : dict with 'subject_id' and 'session_type' arrays
    """
    all_epochs    = []
    subject_ids   = []
    session_types = []

    for rec in recordings:
        data  = rec['data']
        sfreq = rec['sfreq']

        epoch_samples = int(epoch_duration * sfreq)
        step_samples  = int(epoch_samples * (1 - overlap))
        n_samples     = data.shape[1]
        start         = 0

        while start + epoch_samples <= n_samples:
            all_epochs.append(data[:, start:start + epoch_samples])
            subject_ids.append(rec['subject_id'])
            session_types.append(rec['session_type'])
            start += step_samples

    return np.array(all_epochs), {
        'subject_id':   np.array(subject_ids),
        'session_type': np.array(session_types),
    }


def subject_aware_train_test_split(X, features, labels, test_size=0.2, random_state=42):
    """
    Train-test split that keeps all epochs from the same subject together.

    Prevents data leakage: no subject appears in both train and test sets.

    Parameters
    ----------
    X : ndarray (n_epochs, n_channels, n_samples)
        Raw epoched data (not used directly; kept for API consistency).
    features : ndarray (n_epochs, n_features)
        Extracted feature matrix.
    labels : dict
        Output of epoch_recordings — must have 'subject_id' and 'session_type'.
    test_size : float
        Fraction of *subjects* (not epochs) to allocate to the test set.
    random_state : int

    Returns
    -------
    X_train, X_test : ndarray
    y_train, y_test : dict with 'subject_id' and 'session_type' arrays
    """
    subject_ids   = labels['subject_id']
    session_types = labels['session_type']

    unique_subjects = np.unique(subject_ids)
    train_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=test_size, random_state=random_state
    )

    train_mask = np.isin(subject_ids, train_subjects)
    test_mask  = np.isin(subject_ids, test_subjects)

    return (
        features[train_mask],
        features[test_mask],
        {'subject_id': subject_ids[train_mask], 'session_type': session_types[train_mask]},
        {'subject_id': subject_ids[test_mask],  'session_type': session_types[test_mask]},
    )
