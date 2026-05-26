import numpy as np
from scipy import signal


def extract_features(X, sfreq):
    """
    Extract the 5 best features for cognitive load classification:
    1. Frontal Theta-Beta Ratio
    2. Frontal Theta Power
    3. Central Beta Power
    4. Frontal Beta2 Power (15-22 Hz)
    5. Spectral Entropy
    """
    n_epochs, n_channels, n_samples = X.shape
    features = np.zeros((n_epochs, 5))
    
    theta_band = (4, 8)
    beta2_band = (15, 22)
    beta_band = (12, 30)
    
    frontal_left = [0, 2, 4]
    frontal_right = [1, 3, 5]
    central = [8, 9, 17]
    
    for epoch_idx in range(n_epochs):
        frontal_data = X[epoch_idx, frontal_left + frontal_right].mean(axis=0)
        central_data = X[epoch_idx, central].mean(axis=0)
        
        freqs, frontal_psd = signal.welch(frontal_data, sfreq, nperseg=min(256, n_samples))
        _, central_psd = signal.welch(central_data, sfreq, nperseg=min(256, n_samples))
        
        theta_mask = (freqs >= theta_band[0]) & (freqs < theta_band[1])
        beta2_mask = (freqs >= beta2_band[0]) & (freqs < beta2_band[1])
        beta_mask = (freqs >= beta_band[0]) & (freqs < beta_band[1])
        
        frontal_theta = frontal_psd[theta_mask].mean()
        frontal_beta = frontal_psd[beta_mask].mean()
        frontal_beta2 = frontal_psd[beta2_mask].mean()
        central_beta = central_psd[beta_mask].mean()
        
        theta_beta_ratio = (frontal_theta + 1e-8) / (frontal_beta + 1e-8)
        
        combined_psd = np.concatenate([frontal_psd, central_psd])
        psd_norm = combined_psd / (combined_psd.sum() + 1e-10)
        spectral_ent = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        features[epoch_idx] = [
            theta_beta_ratio,
            frontal_theta,
            central_beta,
            frontal_beta2,
            spectral_ent
        ]
    
    return features