import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import butter, filtfilt

SAMPLING_RATE = 400

def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=400, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype="band")
    return filtfilt(b, a, signal)

def process_full_lead(signal):
    ecg_signals, info = nk.ecg_process(signal, sampling_rate=SAMPLING_RATE)
    ecg_signals["ECG_Clean"] = bandpass_filter(ecg_signals["ECG_Raw"], fs=SAMPLING_RATE)
    return ecg_signals


import numpy as np
import pandas as pd

def extract_features(segment_df, sampling_rate=400):
    signal = segment_df["ECG_Clean"].values

    # Get all peak indices
    r_peaks = segment_df.index[segment_df["ECG_R_Peaks"] == 1].values
    p_onsets = segment_df.index[segment_df["ECG_P_Onsets"] == 1].values
    p_offsets = segment_df.index[segment_df["ECG_P_Offsets"] == 1].values
    q_peaks = segment_df.index[segment_df["ECG_Q_Peaks"] == 1].values
    r_onsets = segment_df.index[segment_df["ECG_R_Onsets"] == 1].values
    s_peaks = segment_df.index[segment_df["ECG_S_Peaks"] == 1].values
    t_offsets = segment_df.index[segment_df["ECG_T_Offsets"] == 1].values
    t_onsets = segment_df.index[segment_df["ECG_T_Onsets"] == 1].values

    features = []

    
    for i in range(1, len(r_peaks)):
        r0 = r_peaks[i]
        r_prev = r_peaks[i - 1]
        rr_interval = 0
        if len(r_peaks) >= 2:
            rr_interval = (r0 - r_prev) / sampling_rate

        

        def get_last_before(arr, ref):
            candidates = arr[arr < ref]
            return candidates[-1] if len(candidates) > 0 else None

        def get_first_after(arr, ref):
            candidates = arr[arr > ref]
            return candidates[0] if len(candidates) > 0 else None

        p_onset = get_last_before(p_onsets, r0)
        p_offset = get_first_after(p_offsets, p_onset) if p_onset is not None else None

        q_peak = get_last_before(q_peaks, r0)
        r_onset = get_last_before(r_onsets, r0)
        s_peak = get_first_after(s_peaks, r0)

        t_offset = get_first_after(t_offsets, r0)
        t_onset = None
        if s_peak is not None and t_offset is not None:
            t_onset_candidates = t_onsets[(t_onsets > s_peak) & (t_onsets < t_offset)]
            t_onset = t_onset_candidates[0] if len(t_onset_candidates) > 0 else None

        try:
            feature_dict = {
                "RR_interval_s": rr_interval,
                "Heart_Rate_bpm": 60 / rr_interval if rr_interval > 0 else np.nan,
                "PR_interval_s": (r0 - p_onset) / sampling_rate if p_onset is not None else np.nan,
                "QRS_duration_s": (s_peak - r_onset) / sampling_rate if s_peak is not None and r_onset is not None else np.nan,
                "QT_interval_s": (t_offset - q_peak) / sampling_rate if t_offset is not None and q_peak is not None else np.nan,
                "P_duration_s": (p_offset - p_onset) / sampling_rate if p_onset is not None and p_offset is not None else np.nan,
                "T_duration_s": (t_offset - t_onset) / sampling_rate if t_offset is not None and t_onset is not None else np.nan,
                "P_amplitude": signal[p_onset] if p_onset is not None else np.nan,
                "R_amplitude": signal[r0],
                "T_amplitude": signal[t_onset] if t_onset is not None else np.nan
            }
            features.append(feature_dict)
        except Exception as e:
            print(f"Skipping beat {i} due to error: {e}")
            continue

    df = pd.DataFrame(features)

    return df


def interpret_feature(value, normal_range, label):
    low, high = normal_range
    if np.isnan(value):
        return f"{label}: unavailable"
    status = "normal" if low <= value <= high else "abnormal"
    reasoning = f" (expected {low}–{high}, observed {value:.3f})" if status == "abnormal" else ""
    return f"{label}: {value:.3f} s — {status}{reasoning}"

def interpret_amplitude(amp, normal_amp_range, label):
    low, high = normal_amp_range
    if np.isnan(amp):
        return f"{label}: unavailable"
    status = "normal" if low <= amp <= high else "abnormal"
    reasoning = f" (expected {low}–{high} mV, observed {amp:.3f} mV)" if status=="abnormal" else ""
    return f"{label} amplitude: {amp:.3f} mV — {status}{reasoning}"

def extract_segment(ecg_signals, quality_threshold=0.3):
    
    quality_mask = ecg_signals["ECG_Quality"] >= quality_threshold
    if quality_mask is None:
        print("[WARN] ECG_Quality column missing, recomputing on ECG_Clean")
        qual = nk.ecg_quality(ecg_signals["ECG_Clean"], sampling_rate=SAMPLING_RATE)
        ecg_signals["ECG_Quality"] = qual

    diffs = np.diff(quality_mask.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1

    if quality_mask.iloc[0]:
        starts = np.insert(starts, 0, 0)
    if quality_mask.iloc[-1]:
        ends = np.append(ends, len(quality_mask))

    lengths = ends - starts
    if len(lengths) == 0:
        raise ValueError(f"No segment found with ECG_Quality >= {quality_threshold}")

    longest_idx = np.argmax(lengths)
    start_idx = starts[longest_idx]
    end_idx = ends[longest_idx]

    segment = ecg_signals.iloc[start_idx:end_idx].reset_index(drop=True)
    return segment


def analyze_waves(waves_by_lead, lead_labels):
    report = []
    for i, (signal, lead) in enumerate(zip(waves_by_lead, lead_labels)):
        try:
            ecg_signals = process_full_lead(signal)
            segment = extract_segment(ecg_signals)
            feats = extract_features(segment)

            rr = feats["RR_interval_s"].mean()
            hr = feats["Heart_Rate_bpm"].mean()
            pr = feats["PR_interval_s"].mean()
            qrs = feats["QRS_duration_s"].mean()
            qt = feats["QT_interval_s"].mean()

            r_amp = feats["R_amplitude"].mean()
            q_amp = feats["P_amplitude"].mean() 
            t_amp = feats["T_amplitude"].mean()


            section = [f"**Lead {lead}**"]
            section.append(interpret_feature(rr, (0.6, 1.2), "RR interval"))
            section.append(interpret_feature(pr, (0.12, 0.20), "PR interval"))
            section.append(interpret_feature(qrs, (0.06, 0.10), "QRS duration"))
            section.append(interpret_feature(qt, (0.30, 0.44), "QT interval"))

            section.append(interpret_amplitude(q_amp, (-0.3, -0.05), "Q"))
            section.append(interpret_amplitude(r_amp, (0.5, 1.5), "R"))
            section.append(interpret_amplitude(t_amp, (0.1, 0.5), "T"))

            st = None
            try:
                s_idx = segment.index[segment["ECG_S_Peaks"] == 1].values
                t_onsets = segment.index[segment["ECG_T_Onsets"] == 1].values
                if len(s_idx) > 0 and len(t_onsets) > 0:
                    st_val = segment.loc[t_onsets[0], "ECG_Clean"] - segment.loc[s_idx[0], "ECG_Clean"]
                    st = float(st_val)
            except:
                st = None

            if st is not None and not np.isnan(st):
                thresh = 0.10 if lead in ["II","III","aVF","I","aVL","V4","V5","V6"] else 0.15 if lead=="V3" else 0.20 if lead=="V2" else 0.10
                st_status = "ST-elevated" if st>=thresh else "ST normal"
                section.append(f"ST-segment surrogate: {st:.3f} mV — threshold {thresh:.2f}: {st_status}")

            anomalies = []
            if q_amp < -0.1: anomalies.append("Pathologic Q wave possible")
            if st is not None and st >= thresh: anomalies.append("Suggests STEMI")
            if hr < 50: anomalies.append("Bradycardia")
            if hr > 100: anomalies.append("Tachycardia")
            if not anomalies:
                anomalies.append("No major abnormal findings")

            section.append("🌡️ Interpretation: " + "; ".join(anomalies))
            report.append("\n".join(section))

        except Exception as e:
            print(f"[ERROR] Lead {lead} failed with exception: {e}")
            report.append(f"Lead {lead}: Error — {str(e)}")

    return "\n\n".join(report)