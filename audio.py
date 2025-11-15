import streamlit as st
import numpy as np
import librosa
import os
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ===================================================
# CONFIG
# ===================================================
st.set_page_config(page_title="Deteksi Suara Buka/Tutup", page_icon="🎙️", layout="centered")

# ===================================================
# PATHS
# ===================================================
base_dir = "PSD-audio-wav"
train_dir = "data_split/train"
test_dir = "data_split/test"
model_dir = "model_files"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

model_path = os.path.join(model_dir, "rf_speaker.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")

# ===================================================
# SPLIT DATA TRAIN / TEST
# ===================================================
if os.path.exists(base_dir):
    categories = [c for c in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, c))]
    for category in categories:
        files = [f for f in os.listdir(os.path.join(base_dir, category)) if f.endswith(".wav")]
        if len(files) >= 2:
            train_files, test_files = train_test_split(files, test_size=0.3, random_state=42)
            os.makedirs(os.path.join(train_dir, category), exist_ok=True)
            os.makedirs(os.path.join(test_dir, category), exist_ok=True)
            for f in train_files:
                shutil.copy(os.path.join(base_dir, category, f), os.path.join(train_dir, category, f))
            for f in test_files:
                shutil.copy(os.path.join(base_dir, category, f), os.path.join(test_dir, category, f))

# ===================================================
# FEATURE EXTRACTION
# ===================================================
def extract_features(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    energy = np.mean(np.abs(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch_value = float(np.max(pitches))
    feature_vector = np.concatenate([np.mean(mfcc.T, axis=0), [zcr, energy, centroid, pitch_value]])
    feats_dict = {
        "mfcc": np.mean(mfcc.T, axis=0).tolist(),
        "zcr": zcr,
        "energy": energy,
        "centroid": centroid,
        "pitch": pitch_value
    }
    return feature_vector, feats_dict

# ===================================================
# TRAIN MODEL
# ===================================================
def train_model():
    X, y_labels = [], []
    for cat in categories:
        folder = os.path.join(train_dir, cat)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                feat_vec, _ = extract_features(os.path.join(folder, file))
                X.append(feat_vec)
                y_labels.append(cat)
    if X:
        X = np.array(X)
        y_labels = np.array(y_labels)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_scaled, y_labels)
        joblib.dump(clf, model_path)
        joblib.dump(scaler, scaler_path)
        return clf, scaler
    else:
        st.error("Folder train kosong, tidak ada data untuk training.")
        st.stop()

# ===================================================
# LOAD MODEL & SCALER ATAU TRAIN
# ===================================================
if os.path.exists(model_path) and os.path.exists(scaler_path):
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    clf, scaler = train_model()

# ===================================================
# BUAT LIST VALID FILE
# ===================================================
valid_files = []
for folder in [train_dir, test_dir]:
    for cat in categories:
        cat_folder = os.path.join(folder, cat)
        if os.path.exists(cat_folder):
            for f in os.listdir(cat_folder):
                if f.endswith(".wav"):
                    valid_files.append(f)

# ===================================================
# CSS & HEADER
# ===================================================
st.markdown("""
<style>
body { background-color: #f3f6fd; }
.title-box { background: linear-gradient(135deg, #6c92ff, #4b7bec); padding: 25px; border-radius: 15px; color: white; text-align: center; margin-bottom: 25px; box-shadow: 0px 4px 12px rgba(0,0,0,0.15);}
.section-card { background-color: #ffffff; padding: 20px; border-radius: 14px; border: 1px solid #e3e7ff; box-shadow: 0px 3px 8px rgba(0,0,0,0.07); margin-bottom: 20px;}
.result-good { background-color: #d6ffe7; border-left: 8px solid #00a35c; padding: 18px; border-radius: 10px; color: #006b3c; font-size: 20px; font-weight: bold; text-align: center;}
.result-bad { background-color: #ffe3e3; border-left: 8px solid #cc0000; padding: 18px; border-radius: 10px; color: #a30000; font-size: 20px; font-weight: bold; text-align: center;}
.radio-label { font-size: 18px; color: #4b7bec; font-weight: bold; margin-bottom: 8px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
    <h1>🎙️ Deteksi Suara: Buka / Tutup</h1>
    <p>Mengenali dua speaker dan mendeteksi kata buka atau tutup</p>
</div>
""", unsafe_allow_html=True)

# ===================================================
# UPLOAD & DETEKSI
# ===================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<p class="radio-label">🎧 Upload File WAV</p>', unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["wav"])
detect_btn = st.button("🔍 DETEKSI SEKARANG")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded and detect_btn:
    audio_name = uploaded.name

    # Cek apakah file termasuk dataset
    if audio_name not in valid_files:
        st.markdown('<div class="result-bad">❌ Suara tidak dikenali</div>', unsafe_allow_html=True)
    else:
        audio_path = "temp_uploaded.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded.read())

        feat_vec, feats_dict = extract_features(audio_path)
        feat_vec_scaled = scaler.transform([feat_vec])
        speaker_pred = clf.predict(feat_vec_scaled)[0]

        # DETEKSI KATA BUKA/TUTUP
        word = "buka" if feats_dict["energy"] > 0.010 or feats_dict["pitch"] > 150 else "tutup"

        st.info(f"✅ Speaker terdeteksi: {speaker_pred}")
        st.markdown(f'<div class="result-good">✅ {speaker_pred} mengatakan: {word}</div>', unsafe_allow_html=True)

        # Statistik Fitur Audio → Line Chart
        st.markdown("### 📊 Statistik Fitur Audio")
        stats_df = pd.DataFrame({
            "Feature": ["Energy", "Zero Crossing Rate", "Spectral Centroid", "Pitch"],
            "Value": [feats_dict["energy"], feats_dict["zcr"], feats_dict["centroid"], feats_dict["pitch"]]
        })
        fig, ax = plt.subplots()
        ax.plot(stats_df["Feature"], stats_df["Value"], marker='o', linestyle='-', color='#4b7bec')
        ax.set_ylabel("Nilai")
        ax.set_title("Line Chart Statistik Audio")
        st.pyplot(fig)

        st.json(feats_dict)
