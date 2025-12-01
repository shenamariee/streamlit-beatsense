# app.py
"""
Streamlit app for ECG Arrhythmia Detection — hybrid rule-based + RandomForest
Run locally with:
    pip install -r requirements.txt
    streamlit run app.py
"""
import os
import io
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="ECG Arrhythmia Detection", layout="wide")
st.title("ECG Arrhythmia Detection — Hybrid Rule + ML")
st.markdown("Upload `.hea` and `.dat` (and `.atr` if available). The app will extract beats, train a RF beat classifier, apply sequence-level rules, and optionally train an RF for tachy subtypes.")

WORK_DIR = "ecg_data"
os.makedirs(WORK_DIR, exist_ok=True)

def save_uploaded_files(uploaded_files, dest_dir=WORK_DIR):
    saved = []
    for file in uploaded_files:
        path = os.path.join(dest_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        saved.append(path)
    return saved

def get_base_names(directory):
    basenames = set()
    for fname in os.listdir(directory):
        base, _ = os.path.splitext(fname)
        basenames.add(base)
    return sorted(list(basenames))

def bandpass(sig, fs, low=0.5, high=40):
    b, a = butter(3, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)

def pan_tompkins_detector(signal, fs):
    b, a = butter(3, [5/(fs/2), 15/(fs/2)], btype='band')
    filtered_ecg = filtfilt(b, a, signal)
    diff_signal = np.ediff1d(filtered_ecg, to_end=0)
    squared = diff_signal ** 2
    window_size = int(0.150 * fs)
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    from scipy.signal import find_peaks
    distance = int(0.25 * fs)
    height = np.mean(integrated) * 1.2
    peaks, _ = find_peaks(integrated, distance=distance, height=height)
    refined_peaks = []
    search_radius = int(0.05 * fs)
    for p in peaks:
        start = max(p - search_radius, 0)
        end = min(p + search_radius, len(signal))
        local_max = np.argmax(signal[start:end]) + start
        refined_peaks.append(local_max)
    return np.unique(refined_peaks)

def extract_beats(signal, r_peaks, fs, window_ms=700, resample_len=100):
    half = int((window_ms / 1000) * fs // 2)
    beats = []
    indices = []
    for r in r_peaks:
        if r - half < 0 or r + half >= len(signal):
            continue
        beat = signal[r - half:r + half]
        beats.append(resample(beat, resample_len))
        indices.append(r)
    return np.array(beats), np.array(indices)

def extract_features(beats, rr_intervals):
    features = []
    for i, beat in enumerate(beats):
        rr = rr_intervals[i] if i < len(rr_intervals) else rr_intervals[-1]
        features.append([
            np.mean(beat),
            np.std(beat),
            np.min(beat),
            np.max(beat),
            rr,
            np.median(beat),
            np.percentile(beat, 25),
            np.percentile(beat, 75),
            np.sum(beat**2),
            len(beat)
        ])
    return np.array(features)

def is_irregular(rr_segment, threshold=0.12):
    return np.std(rr_segment) > threshold

def classify_tachycardia_regular(beat_seq):
    if any(b == "V" for b in beat_seq):
        return "Ventricular Tachycardia"
    if any(b == "A" for b in beat_seq):
        return "Atrial Flutter"
    if any(b in ["L", "R"] for b in beat_seq):
        return "Supraventricular Tachycardia"
    return "Supraventricular Tachycardia"

def classify_tachycardia_irregular(beat_seq):
    if any(b == "F" for b in beat_seq):
        return "Atrial Fibrillation"
    if any(b == "V" for b in beat_seq):
        return "Ventricular Fibrillation"
    return "Atrial Fibrillation"

label_map = {"N":0, "L":1, "R":2, "V":3, "A":4, "F":5}
tachy_label_map = {
    "Atrial Fibrillation": 0,
    "Ventricular Tachycardia": 1,
    "Supraventricular Tachycardia": 2,
    "Atrial Flutter": 3,
    "Other Tachy": 4,
    "Not Tachycardia": -1
}

@st.cache_resource
def train_rf_model(X, y, n_estimators=200):
    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', random_state=42)
    clf.fit(X, y)
    return clf

st.sidebar.header("Upload ECG files")
uploaded_files = st.sidebar.file_uploader("Upload .hea, .dat, .atr files (same basename)", type=["hea","dat","atr"], accept_multiple_files=True)
if uploaded_files:
    saved = save_uploaded_files(uploaded_files, dest_dir=WORK_DIR)
    st.sidebar.success(f"Saved {len(saved)} files to {WORK_DIR}")

available_bases = get_base_names(WORK_DIR)
if not available_bases:
    st.info("No files available — upload .hea/.dat/.atr in the sidebar.")
    st.stop()

chosen_base = st.selectbox("Select record (basename)", available_bases)
st.markdown(f"**Selected:** `{chosen_base}`")
max_duration_sec = st.sidebar.number_input("Max duration (sec)", value=120, min_value=10, step=10)
resample_len = st.sidebar.number_input("Beat resample length", value=100, min_value=50, step=10)
window_ms = st.sidebar.number_input("Beat window (ms)", value=700, min_value=300, step=50)
run_button = st.button("Run ECG Analysis")

if run_button:
    st.info("Running analysis...")
    record_path = os.path.join(WORK_DIR, chosen_base)
    try:
        record = wfdb.rdrecord(record_path)
        try:
            ann = wfdb.rdann(record_path, "atr")
            ann_present = True
        except Exception:
            ann = None
            ann_present = False
    except Exception as e:
        st.error(f"Failed to read WFDB record '{chosen_base}': {e}")
        st.stop()

    channels = record.sig_name
    chosen_channel = st.selectbox("Signal channel to use", channels, index=0)
    ch_idx = channels.index(chosen_channel)
    signal = record.p_signal[:, ch_idx]
    fs = record.fs
    st.write(f"fs: {fs} Hz")
    max_samples = int(max_duration_sec * fs)
    signal = signal[:max_samples]

    if ann_present and ann.sample is not None and len(ann.sample) > 0:
        r_peaks = ann.sample
        labels = np.array(ann.symbol) if hasattr(ann, "symbol") else np.array(["N"] * len(r_peaks))
        st.success(f"Annotation found: {len(r_peaks)} annotations.")
    else:
        st.warning("No annotation — running Pan-Tompkins.")
        r_peaks = pan_tompkins_detector(signal, fs)
        labels = np.array(["N"] * len(r_peaks))

    valid_idx = np.where(r_peaks < max_samples)[0]
    r_peaks = r_peaks[valid_idx]
    labels = labels[valid_idx] if len(labels) >= len(valid_idx) else labels[:len(valid_idx)]

    signal_f = bandpass(signal, fs)
    beats, beat_indices = extract_beats(signal_f, r_peaks, fs, window_ms=window_ms, resample_len=resample_len)
    if len(beats) == 0:
        st.error("No beats extracted.")
        st.stop()

    rr = np.diff(r_peaks) / fs
    rr = np.append(rr, rr[-1]) if len(rr)>0 else np.array([1.0])
    beat_features = extract_features(beats, rr)
    y_beats = np.array([label_map.get(l, 0) for l in labels[:len(beat_features)]])

    if len(beat_features) < 5 or len(np.unique(y_beats)) < 2:
        st.warning("Insufficient beat samples/labels for ML. Showing available outputs.")
        clf_beats = None
        pred_beats = np.array([0]*len(y_beats))
    else:
        X_train, X_test, y_train, y_test = train_test_split(beat_features, y_beats, test_size=0.2, random_state=42)
        clf_beats = train_rf_model(X_train, y_train)
        pred_beats = clf_beats.predict(X_test)
        st.subheader("Beat-level classification")
        st.text(classification_report(y_test, pred_beats, zero_division=0))
        st.write("Confusion matrix (beat-level):")
        st.dataframe(pd.DataFrame(confusion_matrix(y_test, pred_beats), index=np.unique(y_test), columns=np.unique(y_test)))

    beat_hr_labels = []
    for rr_val in rr[:len(beat_features)]:
        hr = 60 / rr_val if rr_val > 0 else 0
        if hr < 60:
            beat_hr_labels.append('Bradycardia')
        elif hr > 100:
            beat_hr_labels.append('Tachycardia')
        else:
            beat_hr_labels.append('Normal')

    seq_len = 25
    seq_step = 5
    seq_labels = []
    tachy_results = []
    for i in range(0, max(1, len(rr) - seq_len), seq_step):
        seq_rr = rr[i:i+seq_len]
        if len(seq_rr) == 0:
            continue
        avg_hr = 60 / np.mean(seq_rr) if np.mean(seq_rr) > 0 else 0
        if avg_hr < 60:
            seq_labels.append(0)
            tachy_results.append("Not Tachycardia")
        elif avg_hr > 100:
            seq_labels.append(2)
            seq_beats = labels[i:i+seq_len]
            if is_irregular(seq_rr):
                subtype = classify_tachycardia_irregular(seq_beats)
            else:
                subtype = classify_tachycardia_regular(seq_beats)
            tachy_results.append(subtype)
        else:
            seq_labels.append(1)
            tachy_results.append("Not Tachycardia")

    seq_features = []
    seq_target = []
    seq_index_map = []
    for idx in range(len(seq_labels)):
        start = idx * seq_step
        seq_rr = rr[start : start + seq_len]
        seq_beats = labels[start : start + seq_len]
        if len(seq_rr) < 2:
            continue
        mean_rr = np.mean(seq_rr)
        median_rr = np.median(seq_rr)
        std_rr = np.std(seq_rr)
        rmssd = np.sqrt(np.mean(np.diff(seq_rr)**2)) if len(seq_rr) > 1 else 0.0
        pnn50 = 100.0 * np.sum(np.abs(np.diff(seq_rr)) > 0.05) / max(1, (len(seq_rr)-1))
        avg_hr = 60 / mean_rr if mean_rr > 0 else 0.0
        pause_flag = 1 if np.any(seq_rr > 3.0) else 0
        irregular_flag = 1 if is_irregular(seq_rr) else 0
        total_ann = len(seq_beats) if len(seq_beats) > 0 else 1
        count_V = np.sum([1 for b in seq_beats if b == "V"])
        count_A = np.sum([1 for b in seq_beats if b == "A"])
        count_F = np.sum([1 for b in seq_beats if b == "F"])
        count_LR = np.sum([1 for b in seq_beats if b in ["L", "R"]])
        count_N = np.sum([1 for b in seq_beats if b == "N"])
        percent_V = count_V / total_ann
        percent_A = count_A / total_ann
        percent_F = count_F / total_ann
        percent_LR = count_LR / total_ann
        percent_N = count_N / total_ann
        features = [
            mean_rr, median_rr, std_rr, rmssd, pnn50, avg_hr,
            pause_flag, irregular_flag,
            percent_V, percent_A, percent_F, percent_LR, percent_N
        ]
        seq_features.append(features)
        rule_label = tachy_results[idx]
        mapped = tachy_label_map.get(rule_label, 4)
        seq_target.append(mapped)
        seq_index_map.append(idx)
    seq_features = np.array(seq_features)
    seq_target = np.array(seq_target)
    seq_index_map = np.array(seq_index_map)

    train_mask = seq_target != -1
    X_tachy = seq_features[train_mask]
    y_tachy = seq_target[train_mask]

    use_ml2 = False
    clf_tachy = None
    if len(y_tachy) >= 5 and len(np.unique(y_tachy)) > 1:
        X_tr, X_val, y_tr, y_val = train_test_split(X_tachy, y_tachy, test_size=0.2, random_state=42, stratify=y_tachy if len(np.unique(y_tachy))>1 else None)
        clf_tachy = train_rf_model(X_tr, y_tr)
        use_ml2 = True
        y_pred_val = clf_tachy.predict(X_val)
        st.subheader("Tachycardia subtype classifier (RF #2) validation")
        st.text(classification_report(y_val, y_pred_val, zero_division=0))
        st.write("Confusion matrix (RF #2 validation):")
        st.dataframe(pd.DataFrame(confusion_matrix(y_val, y_pred_val)))
    else:
        st.info("Not enough tachy sequences to train RF #2. Using rule-based subtypes.")

    if use_ml2:
        inv_tachy_map = {v:k for k,v in tachy_label_map.items() if v != -1}
        for row_idx, seq_idx in enumerate(seq_index_map):
            if seq_target[row_idx] != -1:
                pred_int = clf_tachy.predict([seq_features[row_idx]])[0]
                pred_str = inv_tachy_map.get(pred_int, "Other Tachy")
                if seq_labels[seq_idx] == 2:
                    tachy_results[seq_idx] = pred_str

    overall_summary = {"Bradycardia":0, "Normal":0, "Tachycardia":0, "AFib":0, "VT":0, "SVT":0, "AFlutter":0, "Other Tachy":0}
    for i, label in enumerate(seq_labels):
        if label == 0:
            overall_summary["Bradycardia"] += 1
        elif label == 1:
            overall_summary["Normal"] += 1
        else:
            overall_summary["Tachycardia"] += 1
            subtype = tachy_results[i]
            if subtype == "Atrial Fibrillation":
                overall_summary["AFib"] += 1
            elif subtype == "Ventricular Tachycardia":
                overall_summary["VT"] += 1
            elif subtype == "Supraventricular Tachycardia":
                overall_summary["SVT"] += 1
            elif subtype == "Atrial Flutter":
                overall_summary["AFlutter"] += 1
            else:
                overall_summary["Other Tachy"] += 1

    st.subheader("Overall rhythm summary (sequence-level windows)")
    total_sequences = sum(overall_summary.values()) if sum(overall_summary.values())>0 else 1
    summary_table = pd.DataFrame([{"Rhythm Type": k, "Sequences": v, "Percent": (v/total_sequences)*100 if total_sequences>0 else 0.0} for k,v in overall_summary.items() if v>0])
    st.dataframe(summary_table)

    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(signal, label='ECG Signal')
    ax.scatter(r_peaks, signal[r_peaks], color='red', s=10, label='R-peaks')
    ax.set_title(f"ECG Signal (first {len(signal)} samples)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.subheader("ECG plot with detected/annotated R-peaks")
    st.pyplot(fig)

    st.subheader("First 20 beats (ML beat label if available, HR, Sequence-level rhythm)")
    beat_rows = []
    for i in range(min(20, len(beat_features))):
        ml_label = "N/A"
        if clf_beats is not None:
            try:
                ml_pred = clf_beats.predict([beat_features[i]])[0]
                ml_label = [k for k,v in label_map.items() if v==ml_pred][0]
            except Exception:
                ml_label = "N/A"
        hr = 60 / rr[i] if rr[i] > 0 else 0
        seq_idx = min(i, max(0, len(seq_labels)-1))
        if hr < 60:
            beat_rhythm_label = "Bradycardia"
        elif hr > 100:
            beat_rhythm_label = tachy_results[seq_idx] if seq_idx < len(tachy_results) else "Tachycardia"
        else:
            beat_rhythm_label = "Normal"
        beat_rows.append({"Beat": i, "ML_Label": ml_label, "HR_bpm": round(hr,1), "Rhythm": beat_rhythm_label})
    st.table(pd.DataFrame(beat_rows))

    beat_df = pd.DataFrame(beat_features, columns=["mean","std","min","max","rr","median","p25","p75","energy","length"])
    beat_df["annotation_symbol"] = labels[:len(beat_df)]
    beat_df["hr_bpm"] = [60/x if x>0 else 0 for x in beat_df["rr"]]
    sequence_df = pd.DataFrame(seq_features, columns=["mean_rr","median_rr","std_rr","rmssd","pnn50","avg_hr","pause_flag","irregular_flag","percent_V","percent_A","percent_F","percent_LR","percent_N"])
    if len(seq_labels) >= len(sequence_df):
        sequence_df["rule_seq_label"] = seq_labels[:len(sequence_df)]
        sequence_df["rule_tachy_result"] = tachy_results[:len(sequence_df)]
    else:
        sequence_df["rule_seq_label"] = seq_labels + [None]*(len(sequence_df)-len(seq_labels))
        sequence_df["rule_tachy_result"] = tachy_results + [None]*(len(sequence_df)-len(tachy_results))

    st.download_button(label="Download beat-level features (CSV)", data=beat_df.to_csv(index=False).encode('utf-8'), file_name=f"beats_{chosen_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
    st.download_button(label="Download sequence-level summary (CSV)", data=sequence_df.to_csv(index=False).encode('utf-8'), file_name=f"sequences_{chosen_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
    st.success("Analysis complete.")
