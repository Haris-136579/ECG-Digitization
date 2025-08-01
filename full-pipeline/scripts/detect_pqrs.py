import neurokit2 as nk
import numpy as np

def detect_pqrs(signal, sampling_rate=400):
    """
    Detects P, Q, R, S, T peaks in an ECG signal using neurokit2.
    Args:
        signal (np.ndarray): 1D ECG waveform.
        sampling_rate (int): Sampling rate in Hz.
    Returns:
        dict: Dictionary with keys 'P', 'Q', 'R', 'S', 'T', each mapping to an array of peak indices.
        np.ndarray: Cleaned ECG signal (same length as input).
    """
    # Pad to at least 4000 samples for stability (optional, can adjust)
    min_len = 4000
    if len(signal) < min_len:
        signal2 = np.pad(signal, (0, min_len - len(signal)), mode='constant')
    else:
        signal2 = signal

    signals, info = nk.ecg_process(signal2, sampling_rate=sampling_rate)
    cleaned = np.array(signals["ECG_Clean"][:len(signal)])

    peaks_dict = {}
    for wave in ["P", "Q", "R", "S", "T"]:
        peak_mask = np.array(signals[f"ECG_{wave}_Peaks"][:len(signal)])
        peak_indices = np.where(peak_mask == 1)[0]
        peaks_dict[wave] = peak_indices

    return peaks_dict, cleaned