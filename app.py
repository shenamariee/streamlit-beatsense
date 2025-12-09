# app.py
import os
import io
import numpy as np
import pandas as pd
import wfdb
from wfdb import processing
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, resample
from datetime import datetime

st.set_page_config(page_title="BeatSense CNN+LSTM Arrhythmia Classifier", layout="wide")
st.title("BeatSense CNN+LSTM Arrhythmia Classifier")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {DEVICE}")

WORK_DIR = "ecg_data"
os.makedirs(WORK_DIR, exist_ok=True)

# ------------------- Helper functions -------------------
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

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40):
    b, a = butter(3, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
    return filtfilt(b, a, signal)

def baseline_wander_removal(signal, fs):
    # highpass filter at 0.5 Hz to remove baseline wander
    b, a = butter(1, 0.5/(fs/2), btype='high')
    return filtfilt(b, a, signal)

def normalize_beat(beat):
    beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
    return beat

def extract_beats(signal, r_peaks, fs, window_ms=700, resample_len=128):
    half_win = int((window_ms/1000)*fs/2)
    beats = []
    valid_indices = []
    for r in r_peaks:
        if r - half_win < 0 or r + half_win >= len(signal):
            continue
        beat = signal[r - half_win : r + half_win]
        beat_resampled = resample(beat, resample_len)
        beat_norm = normalize_beat(beat_resampled)
        beats.append(beat_norm)
        valid_indices.append(r)
    return np.array(beats), np.array(valid_indices)

# ------------------- Models -------------------

class CNNBeatClassifier(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(32 * 32, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: batch x 1 x 128
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# UPDATED: LSTM for rhythm classification with 7 output classes for tachycardia subtypes
class LSTMRhythmClassifier(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, n_layers=2, n_classes=7):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        # x: batch x seq_len x input_size
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return out

# ------------------- Label mappings -------------------

beat_label_map = {0:"N",1:"L",2:"R",3:"V",4:"A"}  # Example beat classes

# Expanded rhythm classes including tachycardia subtypes
rhythm_label_map = {
    0: "Normal",
    1: "Atrial Fibrillation",
    2: "Atrial Flutter",
    3: "Supraventricular Tachycardia",
    4: "Ventricular Tachycardia",
    5: "Ventricular Fibrillation",
    6: "Other"
}

# ------------------- Load or init models -------------------

@st.cache_resource
def load_models():
    cnn = CNNBeatClassifier(n_classes=len(beat_label_map)).to(DEVICE)
    lstm = LSTMRhythmClassifier(input_size=len(beat_label_map), n_classes=len(rhythm_label_map)).to(DEVICE)
    # TODO: load your pretrained weights here if available
    cnn.eval()
    lstm.eval()
    return cnn, lstm

cnn_model, lstm_model = load_models()

# ------------------- Rule-based tachycardia subtype detection -------------------

def detect_tachycardia_subtype(beat_labels, valid_rpeaks, fs):
    rr_intervals = np.diff(valid_rpeaks) / fs
    heart_rates = 60 / rr_intervals  # bpm
    avg_hr = np.mean(heart_rates) if len(heart_rates) > 0 else 0

    def consecutive_count(labels, target, min_count):
        count = 0
        for lbl in labels:
            if lbl == target:
                count += 1
                if count >= min_count:
                    return True
            else:
                count = 0
        return False

    count_total = len(beat_labels)
    count_v = beat_labels.count('V')
    count_a = beat_labels.count('A')
    count_n = beat_labels.count('N')

    percent_v = count_v / count_total if count_total > 0 else 0
    percent_a = count_a / count_total if count_total > 0 else 0
    percent_n = count_n / count_total if count_total > 0 else 0

    # Rules:

    if percent_v > 0.5 and avg_hr > 150:
        return "Ventricular Fibrillation"

    if consecutive_count(beat_labels, 'V', 3) and avg_hr > 100:
        return "Ventricular Tachycardia"

    if avg_hr > 150 and (percent_n + percent_a) > 0.8:
        return "Supraventricular Tachycardia"

    if percent_a > 0.3 and np.std(rr_intervals) > 0.1:
        return "Atrial Fibrillation"

    if 100 < avg_hr <= 150 and percent_a > 0.4:
        return "Atrial Flutter"

    if avg_hr > 100 and percent_n > 0.8:
        return "Normal (Sinus Tachycardia)"

    return "Normal"

# ------------------- Main app logic -------------------

st.sidebar.header("Upload ECG files")
uploaded_files = st.sidebar.file_uploader("Upload .hea, .dat, (.atr optional) with same basename", type=["hea","dat","atr"], accept_multiple_files=True)

if uploaded_files:
    saved_paths = save_uploaded_files(uploaded_files)
    st.sidebar.success(f"Saved {len(saved_paths)} files")

basenames = get_base_names(WORK_DIR)
if not basenames:
    st.info("Upload ECG record files (.hea + .dat).")
    st.stop()

chosen_base = st.selectbox("Select ECG record", basenames)
max_dur = st.sidebar.number_input("Max duration (seconds)", value=60, min_value=10)
run_button = st.button("Run Analysis")

if run_button:
    record_path = os.path.join(WORK_DIR, chosen_base)
    try:
        record = wfdb.rdrecord(record_path, sampto=int(max_dur*360))  # limit max samples for speed
    except Exception as e:
        st.error(f"Failed to load record: {e}")
        st.stop()

    fs = int(record.fs)
    signal = record.p_signal[:,0]  # first channel default

    # Preprocessing
    signal = baseline_wander_removal(signal, fs)
    signal = bandpass_filter(signal, fs)

    # QRS detection with wfdb gqrs detector
    try:
        r_peaks = processing.gqrs_detect(sig=signal, fs=fs)
    except Exception as e:
        st.warning(f"gqrs detection failed, fallback to Pan-Tompkins")
        from scipy.signal import find_peaks
        r_peaks, _ = find_peaks(signal, distance=fs*0.25, height=np.mean(signal))

    st.write(f"Detected {len(r_peaks)} R-peaks.")

    # Extract beats
    beats, valid_rpeaks = extract_beats(signal, r_peaks, fs)

    if len(beats) == 0:
        st.error("No beats extracted, try adjusting max duration or check input signal.")
        st.stop()

    # Predict beat classes with CNN
    with torch.no_grad():
        inputs = torch.tensor(beats, dtype=torch.float32).unsqueeze(1).to(DEVICE)  # batch x 1 x length
        outputs = cnn_model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()

    beat_labels = [beat_label_map.get(p, "N") for p in preds]

    # Display beat classification summary
    df_beats = pd.DataFrame({
        "Beat Index": np.arange(1, len(beat_labels)+1),
        "R-peak Sample": valid_rpeaks,
        "Label": beat_labels,
    })
    st.subheader("Beat-level classification")
    st.dataframe(df_beats)

    # Prepare sequence input for rhythm classification (one-hot encoded beat labels)
    seq_len = 25
    step = 5
    seq_features = []
    seq_indices = []
    onehot_map = {label:i for i,label in enumerate(beat_label_map.values())}
    onehot_encoded = np.zeros((len(beat_labels), len(beat_label_map)))
    for i, lbl in enumerate(beat_labels):
        if lbl in onehot_map:
            onehot_encoded[i, onehot_map[lbl]] = 1
    for start in range(0, len(beat_labels)-seq_len, step):
        seq = onehot_encoded[start:start+seq_len]
        seq_features.append(seq)
        seq_indices.append(start)

    if len(seq_features) == 0:
        st.warning("Not enough beats for sequence-level rhythm classification.")
        st.stop()

    seq_features_tensor = torch.tensor(seq_features, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        seq_outputs = lstm_model(seq_features_tensor)
        seq_probs = torch.softmax(seq_outputs, dim=1)
        seq_preds = torch.argmax(seq_probs, dim=1).cpu().numpy()

    lstm_rhythm_labels = [rhythm_label_map.get(p, "Other") for p in seq_preds]

    # Display sequence-level rhythm classification (LSTM)
    df_seq = pd.DataFrame({
        "Sequence Start Beat": seq_indices,
        "Sequence End Beat": [i+seq_len for i in seq_indices],
        "LSTM Rhythm Classification": lstm_rhythm_labels,
    })
    st.subheader("Sequence-level rhythm classification (LSTM)")
    st.dataframe(df_seq)

    # Rule-based tachycardia subtype detection
    rule_rhythm_label = detect_tachycardia_subtype(beat_labels, valid_rpeaks, fs)
    st.subheader("Rule-based Tachycardia subtype detection")
    st.write(f"Detected rhythm subtype: **{rule_rhythm_label}**")

    # Plot ECG with R-peaks highlighted
    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(signal, label="Filtered ECG")
    ax.scatter(valid_rpeaks, signal[valid_rpeaks], color='red', s=10, label="R-peaks")
    ax.set_title("ECG signal with detected R-peaks")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

    # Optionally download results
    csv_beats = df_beats.to_csv(index=False).encode('utf-8')
    st.download_button("Download Beat-level classification CSV", csv_beats, file_name=f"beat_labels_{chosen_base}.csv")

    csv_seq = df_seq.to_csv(index=False).encode('utf-8')
    st.download_button("Download Sequence-level rhythm classification CSV", csv_seq, file_name=f"rhythm_seq_{chosen_base}.csv")

else:
    st.info("Upload ECG record files (.hea + .dat) and click 'Run Analysis'.")
