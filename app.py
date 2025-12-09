# app.py
"""
Streamlit app for BeatSense: Python-Based Arrhythmia Detection Through Signal Processing
Run locally with:
    pip install -r requirements.txt
    streamlit run app.py
"""
import sys
print(f"Python version: {sys.version}")

import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, resample
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="BeatSense", layout="wide")
st.title("BeatSense: Python-Based Arrhythmia Detection Through Signal Processing")

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
    "Tachycardia": -1
}

# CNN model
class ECGBeatCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(ECGBeatCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64 * (resample_len // 2), 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

@st.cache_resource
def load_cnn_model(model_path=None, num_classes=6):
    model = ECGBeatCNN(num_classes=num_classes)
    if model_path is not None and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_beats_cnn(model, beat_segments):
    inputs = torch.tensor(beat_segments, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).numpy()
    return preds

# UI and main pipeline
st.sidebar.header("Upload ECG files")
uploaded_files = st.sidebar.file_uploader("Upload .hea, .dat, .atr files (same basename)", type=["hea","dat","atr"], accept_multiple_files=True)
if uploaded_files:
    save_uploaded_files(uploaded_files, dest_dir=WORK_DIR)
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
    st.write(f"Sampling frequency: {fs} Hz")
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
    y_beats = np.array([label_map.get(l, 0) for l in labels[:len(beats)]])

    model_path = os.path.join(WORK_DIR, "cnn_beat_classifier.pth")
    cnn_model = load_cnn_model(model_path=model_path, num_classes=len(label_map))

    beats_norm = (beats - np.mean(beats, axis=1, keepdims=True)) / (np.std(beats, axis=1, keepdims=True) + 1e-6)

    if len(beats_norm) < 5 or len(np.unique(y_beats)) < 2:
        st.warning("Insufficient beat samples/labels for ML. Showing available outputs.")
        cnn_preds = np.zeros(len(beats_norm), dtype=int)
    else:
        cnn_preds = predict_beats_cnn(cnn_model, beats_norm)

    inv_label_map = {v:k for k,v in label_map.items()}
    beat_predictions = [inv_label_map.get(p, "N/A") for p in cnn_preds]

    # Sequence window params
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
            tachy_results.append("Bradycardia")
        elif avg_hr > 100:
            seq_labels.append(2)
            seq_beats = beat_predictions[i:i+seq_len]
            if is_irregular(seq_rr):
                subtype = classify_tachycardia_irregular(seq_beats)
            else:
                subtype = classify_tachycardia_regular(seq_beats)
            tachy_results.append(subtype)
        else:
            seq_labels.append(1)
            tachy_results.append("Normal")

    seq_features = []
    seq_target = []
    seq_index_map = []
    for idx in range(len(seq_labels)):
        start = idx * seq_step
        seq_rr = rr[start : start + seq_len]
        seq_beats = beat_predictions[start : start + seq_len]
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

    if len(seq_features) < 5 or len(np.unique(seq_target)) < 2:
        st.warning("Insufficient sequence samples for RF classifier. Displaying sequence-level rule results only.")
        seq_pred_labels = seq_target
    else:
        rf_clf = RandomForestClassifier(n_estimators=150, random_state=42)
        rf_clf.fit(seq_features, seq_target)
        seq_pred_labels = rf_clf.predict(seq_features)

    # Overall summary
    overall_summary = {"Bradycardia":0, "Normal":0, "Tachycardia":0, "AFib":0, "VT":0, "SVT":0, "AFlutter":0, "Other Tachy":0}
    for i, label in enumerate(seq_pred_labels):
        if label == 0:
            overall_summary["Bradycardia"] += 1
        elif label == 1:
            overall_summary["Normal"] += 1
        else:
            overall_summary["Tachycardia"] += 1
            orig_label = list(tachy_label_map.keys())[list(tachy_label_map.values()).index(label)]
            if orig_label == "Atrial Fibrillation":
                overall_summary["AFib"] += 1
            elif orig_label == "Ventricular Tachycardia":
                overall_summary["VT"] += 1
            elif orig_label == "Supraventricular Tachycardia":
                overall_summary["SVT"] += 1
            elif orig_label == "Atrial Flutter":
                overall_summary["AFlutter"] += 1
            else:
                overall_summary["Other Tachy"] += 1

    st.subheader("Overall rhythm summary (sequence-level windows)")
    total_sequences = sum(overall_summary.values()) if sum(overall_summary.values())>0 else 1
    summary_table = pd.DataFrame([{"Rhythm Type": k, "Sequences": v, "Percent": (v/total_sequences)*100} for k,v in overall_summary.items() if v>0])
    st.dataframe(summary_table)

    # Plot ECG + R peaks
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(signal, label='ECG Signal')
    ax.scatter(r_peaks, signal[r_peaks], color='red', s=10, label='R-peaks')
    ax.set_title(f"ECG Signal (first {len(signal)} samples)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.subheader("ECG plot with detected/annotated R-peaks")
    st.pyplot(fig)

    st.subheader("First 20 beats (CNN beat label)")
    beat_table = pd.DataFrame({
        "Beat Index": np.arange(min(20, len(beat_predictions))),
        "Predicted Beat Label": beat_predictions[:20],
        "Original Label": labels[:20],
        "RR Interval (s)": rr[:20]
    })
    st.dataframe(beat_table)
