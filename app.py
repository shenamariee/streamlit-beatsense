# app.py
"""
BeatSense — improved Streamlit app for arrhythmia/rhythm detection.
Features:
 - Robust R-peak detection: neurokit2 (if installed) else improved Pan-Tompkins
 - Strict annotation alignment
 - Rich beat-level + sequence-level features (including optional wavelets)
 - Safe train/test splitting with stratify fallback
 - Optionally uses SMOTE if imbalanced-learn is installed
 - Pipeline with scaler + classifier, stratified CV evaluation
"""
import sys, os
print(f"Python version: {sys.version}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

# signal processing
from scipy.signal import butter, filtfilt, resample, find_peaks
import wfdb

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Optional libs (graceful)
try:
    import neurokit2 as nk
    HAS_NK = True
except Exception:
    HAS_NK = False

try:
    import pywt
    HAS_PYWT = True
except Exception:
    HAS_PYWT = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

st.set_page_config(page_title="BeatSense (improved)", layout="wide")
st.title("BeatSense (improved): Arrhythmia & Rhythm Classification")
st.markdown("Upload `.hea` + `.dat` (and `.atr`), choose options, and run analysis. Optional libs: neurokit2, pywt, imbalanced-learn, xgboost (improves accuracy).")

WORK_DIR = "ecg_data"
os.makedirs(WORK_DIR, exist_ok=True)

# -------------------------
# Utilities & detectors
# -------------------------
def save_uploaded_files(uploaded_files, dest_dir=WORK_DIR):
    saved = []
    for file in uploaded_files:
        path = os.path.join(dest_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        saved.append(path)
    return saved

def bandpass(sig, fs, low=0.5, high=40):
    b, a = butter(3, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, sig)

def pan_tompkins_detector(signal, fs):
    # improved pan-tompkins style detector (robust defaults)
    b, a = butter(3, [5/(fs/2), 15/(fs/2)], btype='band')
    filtered = filtfilt(b,a,signal)
    diff = np.ediff1d(filtered, to_end=0)
    squared = diff**2
    win = max(1, int(0.150*fs))
    integrated = np.convolve(squared, np.ones(win)/win, mode='same')
    distance = max(1, int(0.25*fs))
    height = max(1e-8, np.mean(integrated)*0.7)
    peaks, props = find_peaks(integrated, distance=distance, height=height)
    # refine to actual signal maxima within window
    refined = []
    search_radius = max(1, int(0.05*fs))
    for p in peaks:
        s = max(0, p-search_radius); e = min(len(signal), p+search_radius+1)
        local = np.argmax(signal[s:e]) + s
        refined.append(int(local))
    refined = np.array(sorted(set(refined)), dtype=int)
    return refined

def detect_r_peaks(signal, fs, method="auto"):
    """
    returns numpy array of r_peak indices (integers).
    method: 'neurokit' or 'pan' or 'auto'
    """
    if method == "neurokit" and HAS_NK:
        try:
            out = nk.ecg_peaks(signal, sampling_rate=fs)
            peaks = np.where(out['ECG_R_Peaks'].values==1)[0]
            return np.array(peaks, dtype=int)
        except Exception:
            pass
    if method == "auto" and HAS_NK:
        try:
            out = nk.ecg_peaks(signal, sampling_rate=fs)
            peaks = np.where(out['ECG_R_Peaks'].values==1)[0]
            return np.array(peaks, dtype=int)
        except Exception:
            return pan_tompkins_detector(signal, fs)
    return pan_tompkins_detector(signal, fs)

# -------------------------
# Feature extraction
# -------------------------
def extract_beats(signal, r_peaks, fs, window_ms=700, resample_len=100):
    half = int(((window_ms/1000.0)*fs)/2.0)
    beats = []
    indices = []
    for r in r_peaks:
        r = int(r)
        if r-half < 0 or r+half >= len(signal):
            continue
        beat = signal[r-half:r+half]
        beats.append(resample(beat, resample_len))
        indices.append(r)
    if len(beats)==0:
        return np.empty((0,resample_len)), np.array([], dtype=int)
    return np.array(beats), np.array(indices, dtype=int)

def beat_morph_features(beat, fs, resample_len=100):
    # simple morphological features from beat array (resampled)
    b = np.array(beat)
    first = np.gradient(b)
    second = np.gradient(first)
    energy = np.sum(b**2)
    peak = np.max(b)
    trough = np.min(b)
    mean = np.mean(b); std = np.std(b)
    qrs_width_est = estimate_qrs_width(b, fs, resample_len)
    slope_max = np.max(first); slope_std = np.std(first)
    # PCA features (first 3 components)
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pc = pca.fit_transform(b.reshape(1,-1)).reshape(-1)
    except Exception:
        pc = np.zeros(3)
    feats = [mean, std, np.min(b), np.max(b), energy, peak, trough, slope_max, slope_std, qrs_width_est] + pc.tolist()
    return np.array(feats)

def estimate_qrs_width(beat_resampled, fs, resample_len):
    # crude QRS width: measure contiguous region around max that is above 30% of (max-min)
    b = beat_resampled
    mx = np.max(b); mn = np.min(b)
    thresh = mn + 0.3*(mx-mn)
    above = np.where(b >= thresh)[0]
    if len(above)==0:
        return 0.0
    width_samples = above[-1] - above[0] + 1
    # convert to ms: original window_ms/resample_len gives ms per sample
    # compute original ms per sample: assume window length is ((resample_len/fs)*1000) roughly,
    # but we used resample_len, so approximate width_ms = width_samples * (window_ms/resample_len)
    # avoid dependency on window_ms here; return normalized width in samples
    return float(width_samples)

def wavelet_features(beat):
    # requires pywt
    if not HAS_PYWT:
        return np.zeros(6)
    coeffs = pywt.wavedec(beat, 'db4', level=3)
    feats = []
    for c in coeffs:
        feats.append(np.mean(np.abs(c)))
        feats.append(np.std(c))
    return np.array(feats[:6])

def build_beat_feature_matrix(beats, rr_intervals, fs, resample_len=100):
    X = []
    for i, b in enumerate(beats):
        rr = rr_intervals[i] if i < len(rr_intervals) else rr_intervals[-1] if len(rr_intervals)>0 else 1.0
        morph = beat_morph_features(b, fs, resample_len)
        wf = wavelet_features(b)
        feat = np.concatenate(([rr], morph, wf))
        X.append(feat)
    return np.array(X)

# -------------------------
# Streamlit UI
# -------------------------
st.sidebar.header("Upload ECG files")
uploaded_files = st.sidebar.file_uploader("Upload .hea, .dat, .atr files (same basename)", type=["hea","dat","atr"], accept_multiple_files=True)
if uploaded_files:
    saved = save_uploaded_files(uploaded_files, dest_dir=WORK_DIR)
    st.sidebar.success(f"Saved {len(saved)} files to {WORK_DIR}")

available_bases = sorted({os.path.splitext(f)[0] for f in os.listdir(WORK_DIR)}) if os.path.exists(WORK_DIR) else []
if not available_bases:
    st.info("No files available — upload .hea/.dat/.atr in the sidebar.")
    st.stop()

chosen_base = st.selectbox("Select record (basename)", available_bases)
st.markdown(f"**Selected:** `{chosen_base}`")
max_duration_sec = st.sidebar.number_input("Max duration (sec)", value=120, min_value=10, step=10)
resample_len = st.sidebar.number_input("Beat resample length", value=100, min_value=50, step=10)
window_ms = st.sidebar.number_input("Beat window (ms)", value=700, min_value=300, step=50)
detector_choice = st.sidebar.selectbox("R-peak detector (if available)", ["auto", "neurokit", "pan_tompkins"])
use_smote = st.sidebar.checkbox("Use SMOTE (if imbalanced-learn installed)", value=HAS_SMOTE)
classifier_choice = st.sidebar.selectbox("Classifier", ["RandomForest", "XGBoost (if available)"])
run_button = st.button("Run ECG Analysis")

# -------------------------
# Main pipeline
# -------------------------
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
    raw_signal = record.p_signal[:, ch_idx].astype(float)
    fs = float(record.fs)
    st.write(f"fs: {fs} Hz")
    max_samples = int(max_duration_sec*fs)
    signal = raw_signal[:max_samples]

    # annotations => r_peaks if available
    if ann_present and hasattr(ann, "sample") and len(ann.sample)>0:
        ann_samples = np.array(ann.sample, dtype=int)
        mask = ann_samples < max_samples
        ann_samples = ann_samples[mask]
        ann_symbols = np.array(ann.symbol)[mask] if hasattr(ann, "symbol") else np.array(["N"]*len(ann_samples))
        if len(ann_samples)>0:
            st.success(f"Annotation found: {len(ann_samples)} annotations.")
            r_peaks = ann_samples
            labels = ann_symbols
            ann_present = True
        else:
            st.warning("Annotations exist but none within the selected time window.")
            ann_present = False

    if not ann_present:
        st.warning("No usable annotation — running peak detector.")
        method = detector_choice
        if method == "neurokit":
            method = "neurokit"
        elif method == "pan_tompkins":
            method = "pan"
        else:
            method = "auto"
        r_peaks = detect_r_peaks(signal, int(fs), method=method)
        labels = np.array(["N"]*len(r_peaks))

    # filter r_peaks to within window
    r_peaks = np.array([int(x) for x in r_peaks if 0 <= int(x) < max_samples], dtype=int)
    if len(r_peaks) == 0:
        st.error("No R-peaks after filtering. Try a different detector or increase duration.")
        st.stop()

    # align labels to r_peaks if ann present but lengths mismatch
    if ann_present and len(labels) != len(r_peaks):
        try:
            ann_samples = np.array(ann.sample)
            ann_symbols = np.array(ann.symbol) if hasattr(ann, "symbol") else np.array(["N"]*len(ann_samples))
            aligned = []
            for rp in r_peaks:
                idx = np.argmin(np.abs(ann_samples - rp))
                aligned.append(ann_symbols[idx])
            labels = np.array(aligned)
        except Exception:
            labels = np.array(["N"]*len(r_peaks))

    # signal filtering and beat extraction
    signal_f = bandpass(signal, fs)
    beats, beat_indices = extract_beats(signal_f, r_peaks, fs, window_ms=window_ms, resample_len=int(resample_len))
    if len(beats)==0:
        st.error("No beats extracted (window may be too large or R-peaks near boundaries).")
        st.stop()

    # RR (in seconds) aligned to beats (use diff on r_peaks)
    rr = np.diff(r_peaks)/fs
    if len(rr)==0:
        rr = np.array([1.0])
    else:
        rr = np.append(rr, rr[-1])

    # labels -> numeric
    label_map = {"N":0, "L":1, "R":2, "V":3, "A":4, "F":5}
    y_beats = np.array([label_map.get(s, 0) for s in labels[:len(beats)]])

    # build beat-level feature matrix (rich features)
    beat_X = build_beat_feature_matrix(beats, rr, fs, resample_len=int(resample_len))
    # sanity shapes
    st.write(f"Beats extracted: {len(beats)}, feature dim: {beat_X.shape[1]}")

    # handle tiny datasets
    if len(beat_X) < 5 or len(np.unique(y_beats)) < 2:
        st.warning("Insufficient beats/classes for ML training. Showing available outputs only.")
        clf_beats = None
    else:
        # Safe stratify logic (disable if a class has <2)
        unique, counts = np.unique(y_beats, return_counts=True)
        if (counts < 2).any():
            strat = None
            st.warning("Stratify disabled: some classes have <2 examples.")
        else:
            strat = y_beats

        # optionally apply SMOTE (only if installed)
        X_train, X_test, y_train, y_test = train_test_split(beat_X, y_beats, test_size=0.2, random_state=42, stratify=strat)

        if use_smote and HAS_SMOTE:
            try:
                sm = SMOTE(random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)
                st.info("SMOTE applied to training set.")
            except Exception as e:
                st.warning(f"SMOTE failed: {e}")

        # classifier selection
        if classifier_choice == "XGBoost (if available)" and HAS_XGB:
            clf_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
        else:
            clf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf_model)
        ])
        pipeline.fit(X_train, y_train)
        clf_beats = pipeline
        y_pred = pipeline.predict(X_test)
        st.subheader("Beat-level classification")
        st.text(classification_report(y_test, y_pred, zero_division=0))
        st.write("Confusion matrix (beat-level):")
        st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred), index=np.unique(y_test), columns=np.unique(y_test)))

        # cross-validation (stratified)
        try:
            skf = StratifiedKFold(n_splits=min(5, max(2, len(np.unique(y_beats)))), shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipeline, beat_X, y_beats, cv=skf, scoring='f1_macro')
            st.info(f"Stratified CV F1-macro: mean={np.mean(cv_scores):.3f}, std={np.std(cv_scores):.3f}")
        except Exception as e:
            st.warning(f"CV failed: {e}")

    # Construct sequence-level windows for HRV and rhythm detection (sliding)
    seq_len = 25
    seq_step = 5
    seq_labels = []
    tachy_results = []
    seq_windows = []
    for start in range(0, max(1, len(rr)), seq_step):
        seq_rr = rr[start:start+seq_len]
        if len(seq_rr)==0:
            continue
        avg_hr = 60/np.mean(seq_rr) if np.mean(seq_rr)>0 else 0.0
        if avg_hr < 60:
            seq_labels.append(0); tachy_results.append("Bradycardia")
        elif avg_hr > 100:
            seq_labels.append(2)
            seq_beats = labels[start:start+seq_len]
            if is_irreg := (np.std(seq_rr) > 0.12):
                subtype = "Atrial Fibrillation" if ("F" in seq_beats) else "Ventricular Tachycardia" if ("V" in seq_beats) else "Atrial Fibrillation"
            else:
                subtype = "Ventricular Tachycardia" if ("V" in seq_beats) else "Supraventricular Tachycardia"
            tachy_results.append(subtype)
        else:
            seq_labels.append(1); tachy_results.append("Normal")
        seq_windows.append((start, start+len(seq_rr)))

    # build sequence features for potential subtype RF training (same as earlier, but consistent)
    seq_features = []
    seq_target  = []
    for idx, (start, end) in enumerate(seq_windows):
        seq_rr = rr[start:end]
        seq_beats = labels[start:end]
        if len(seq_rr) < 2:
            continue
        mean_rr = np.mean(seq_rr); median_rr = np.median(seq_rr); std_rr = np.std(seq_rr)
        rmssd = np.sqrt(np.mean(np.diff(seq_rr)**2)) if len(seq_rr)>1 else 0.0
        pnn50 = 100.0 * np.sum(np.abs(np.diff(seq_rr))>0.05) / max(1, (len(seq_rr)-1))
        avg_hr = 60/mean_rr if mean_rr>0 else 0.0
        pause_flag = int(np.any(seq_rr > 3.0))
        irregular_flag = int(std_rr > 0.12)
        total = len(seq_beats) if len(seq_beats)>0 else 1
        count_V = np.sum([1 for b in seq_beats if b=="V"])
        count_A = np.sum([1 for b in seq_beats if b=="A"])
        count_F = np.sum([1 for b in seq_beats if b=="F"])
        count_LR = np.sum([1 for b in seq_beats if b in ["L","R"]])
        count_N = np.sum([1 for b in seq_beats if b=="N"])
        features = [mean_rr, median_rr, std_rr, rmssd, pnn50, avg_hr, pause_flag, irregular_flag,
                    count_V/total, count_A/total, count_F/total, count_LR/total, count_N/total]
        seq_features.append(features)
        # map rule -> label (same mapping as before)
        rule = tachy_results[idx]
        map_dict = {"Atrial Fibrillation":0, "Ventricular Tachycardia":1, "Supraventricular Tachycardia":2, "Atrial Flutter":3, "Other Tachy":4, "Normal":-1, "Bradycardia":-1}
        seq_target.append(map_dict.get(rule, 4))

    seq_features = np.array(seq_features)
    seq_target = np.array(seq_target)

    # If enough sequences, train RF for subtype
    try:
        if len(seq_features)>0 and np.sum(seq_target!=-1) >= 5 and len(np.unique(seq_target[seq_target!=-1]))>1:
            mask = seq_target!=-1
            Xs = seq_features[mask]; ys = seq_target[mask]
            # safe stratify check
            u,c = np.unique(ys, return_counts=True)
            strat2 = ys if (c>=2).all() else None
            Xtr, Xval, ytr, yval = train_test_split(Xs, ys, test_size=0.2, random_state=42, stratify=strat2)
            clf2 = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))])
            clf2.fit(Xtr, ytr)
            ypred_val = clf2.predict(Xval)
            st.subheader("Tachycardia subtype validation (RF #2)")
            st.text(classification_report(yval, ypred_val, zero_division=0))
        else:
            st.info("Not enough labeled tachy sequences to train subtype RF. Using rule-based labels.")
            clf2 = None
    except Exception as e:
        st.warning(f"Subtype RF training failed: {e}")
        clf2 = None

    # overall summary (counts)
    overall = {"Bradycardia":0, "Normal":0, "Tachycardia":0, "AFib":0, "VT":0, "SVT":0, "AFlutter":0, "Other Tachy":0}
    for i,label in enumerate(seq_labels):
        if label==0: overall["Bradycardia"] +=1
        elif label==1: overall["Normal"] +=1
        else:
            overall["Tachycardia"]+=1
            stype = tachy_results[i]
            if stype=="Atrial Fibrillation": overall["AFib"]+=1
            elif stype=="Ventricular Tachycardia": overall["VT"]+=1
            elif stype=="Supraventricular Tachycardia": overall["SVT"]+=1
            elif stype=="Atrial Flutter": overall["AFlutter"]+=1
            else: overall["Other Tachy"]+=1

    st.subheader("Sequence summary")
    total_seq = sum(overall.values()) if sum(overall.values())>0 else 1
    summary_df = pd.DataFrame([{"Rhythm Type":k,"Sequences":v,"Percent":(v/total_seq)*100} for k,v in overall.items() if v>0])
    st.dataframe(summary_df)

    # plot ECG with peaks
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(signal, label='ECG')
    ax.scatter(r_peaks, signal[r_peaks], color='red', s=10, label='R-peaks')
    ax.set_title("ECG with detected R-peaks")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

    # first 20 beats table
    st.subheader("First 20 beats (ML label if available, HR, Sequence-level rhythm)")
    rows=[]
    for i in range(min(20, len(beat_X))):
        ml_label="N/A"
        if 'clf_beats' in locals() and clf_beats is not None:
            try:
                pred = clf_beats.predict([beat_X[i]])[0]
                rev_map = {v:k for k,v in label_map.items()}
                ml_label = rev_map.get(pred, str(pred))
            except Exception:
                ml_label="N/A"
        hr = 60/rr[i] if i < len(rr) and rr[i]>0 else 0
        # find seq index containing beat index i
        seq_idx = None
        for j,(s,e) in enumerate(seq_windows):
            if i>=s and i<e:
                seq_idx=j; break
        if seq_idx is None: seq_idx = max(0, len(seq_labels)-1)
        if hr<60: rhythm_label="Bradycardia"
        elif hr>100: rhythm_label = tachy_results[seq_idx] if seq_idx < len(tachy_results) else "Tachycardia"
        else: rhythm_label="Normal"
        rows.append({"Beat":i,"ML_Label":ml_label,"HR_bpm":round(hr,1),"Rhythm":rhythm_label})
    st.table(pd.DataFrame(rows))

    # downloadable CSVs
    beat_df = pd.DataFrame(beat_X)
    beat_df["annotation_symbol"] = labels[:len(beat_df)]
    beat_df["hr_bpm"] = [60/x if x>0 else 0 for x in rr[:len(beat_df)]]
    seq_df = pd.DataFrame(seq_features, columns=["mean_rr","median_rr","std_rr","rmssd","pnn50","avg_hr","pause_flag","irregular_flag","pct_V","pct_A","pct_F","pct_LR","pct_N"])
    st.download_button("Download beat features", data=beat_df.to_csv(index=False).encode(), file_name=f"beats_{chosen_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    st.download_button("Download seq features", data=seq_df.to_csv(index=False).encode(), file_name=f"seq_{chosen_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    st.success("Analysis complete.")
