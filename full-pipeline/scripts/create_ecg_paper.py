import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend first
import matplotlib.pyplot as plt  # Now import pyplot

import yaml
from scripts.detect_pqrs import detect_pqrs

# Load left and right labels from YAML config
with open('./configs/lead_segmentation.yaml', 'r') as f:
    config = yaml.safe_load(f)
LEFT_LABELS = config['left_labels']
RIGHT_LABELS = config['right_labels']

def create_ecg_paper(leads, labels, output_path, fs=400):
    """
    Creates a 6x2 ECG paper plot with real grid and individual waveforms.

    Args:
        leads (list of np.ndarray): List of waveforms (1D arrays).
        labels (list of str): Corresponding lead names.
        output_path (str): Path to save the image.
        fs (int): Sampling frequency in Hz.
    """
    assert len(leads) == len(labels)
    n_rows, n_cols = 6, 2
    assert len(LEFT_LABELS) == n_rows and len(RIGHT_LABELS) == n_rows, "Each column must have 6 leads"

    # Map label to lead
    label_to_lead = {label: lead for lead, label in zip(leads, labels)}

    # Arrange leads: left column (top to bottom), right column (top to bottom)
    ordered_leads = []
    ordered_labels = []
    for i in range(n_rows):
        # Left column
        if LEFT_LABELS[i] in label_to_lead:
            ordered_leads.append(label_to_lead[LEFT_LABELS[i]])
            ordered_labels.append(LEFT_LABELS[i])
        else:
            ordered_leads.append(np.zeros_like(leads[0]))  # blank if missing
            ordered_labels.append(LEFT_LABELS[i])
        # Right column
        if RIGHT_LABELS[i] in label_to_lead:
            ordered_leads.append(label_to_lead[RIGHT_LABELS[i]])
            ordered_labels.append(RIGHT_LABELS[i])
        else:
            ordered_leads.append(np.zeros_like(leads[0]))  # blank if missing
            ordered_labels.append(RIGHT_LABELS[i])

    # Time axis
    durations = [len(lead) / fs for lead in ordered_leads]
    max_duration = max(durations)
    time = np.arange(int(max_duration * fs)) / fs

    # ECG paper constants
    sec_per_big_square = 0.2
    mv_per_big_square = 0.5
    small_sec = 0.04
    small_mv = 0.1

    amplitude_scale = 10.0

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 14), sharex=True)
    axes = axes.flatten()

    for i, (lead, label) in enumerate(zip(ordered_leads, ordered_labels)):
        ax = axes[i]
        padded = np.full_like(time, np.nan)
        padded[:len(lead)] = lead * amplitude_scale

        # Try to detect peaks and cleaned signal
        try:
            peaks, cleaned = detect_pqrs(lead, sampling_rate=fs)
            cleaned_stretched = cleaned * amplitude_scale
        except Exception as e:
            print(f"[Warning] Failed to detect PQRST in lead '{label}': {e}")
            cleaned = lead
            peaks = {}
            cleaned_stretched = cleaned * amplitude_scale

        # --- Grid drawing ---
        ax.set_facecolor('white')
        ax.set_xlim(0, max_duration)
        y_margin = 1.5
        ax.set_ylim(np.nanmin(cleaned_stretched) - y_margin, np.nanmax(cleaned_stretched) + y_margin)

        # Draw vertical grid lines (time)
        for x in np.arange(0, max_duration, small_sec):
            ax.axvline(x, color='#f8cccc', linewidth=0.5 if x % sec_per_big_square else 1.0, zorder=0)
        # Draw horizontal grid lines (voltage)
        y_min, y_max = -2, 2
        for y in np.arange(np.floor(y_min), np.ceil(y_max), small_mv):
            ax.axhline(y, color='#f8cccc', linewidth=0.5 if abs(y) % mv_per_big_square < 1e-6 else 1.0, zorder=0)

        # Plot ECG
        ax.plot(time[:len(cleaned)], cleaned_stretched, color='black', linewidth=1.2)

        # Plot peaks only if available
        colors = {"P": "green", "Q": "red", "R": "purple", "S": "orange", "T": "blue"}
        for wave, color in colors.items():
            if wave in peaks and isinstance(peaks[wave], np.ndarray):
                ax.plot(time[peaks[wave]], cleaned_stretched[peaks[wave]], 'o', color=color, markersize=6)

        # Lead label
        ax.text(0.01, 0.85, label, fontsize=13, color='darkred', weight='bold', transform=ax.transAxes)

        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=(i >= (n_rows * n_cols) - n_cols))

    # Set consistent y-limits across all subplots
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    fig.suptitle("6x2 ECG Paper (Each Lead on Real Grid)", fontsize=20)
    fig.supxlabel("Time (s)", fontsize=15)
    fig.supylabel("Voltage (mV)", fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
