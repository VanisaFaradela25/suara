import streamlit as st
import numpy as np
import librosa
import os
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
import io

# ===================================================
# CONFIG
# ===================================================
st.set_page_config(
    page_title="Deteksi Suara Buka/Tutup",
    page_icon="🎙️",
    layout="centered"
)

# ===================================================
# CSS
# ===================================================
st.markdown("""
<style>
body { background-color: #f3f6fd; }

.title-box {
    background: linear-gradient(135deg, #6c92ff, #4b7bec);
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}

.section-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e3e7ff;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.07);
    margin-bottom: 20px;
}

.result-good {
    background-color: #d6ffe7;
    border-left: 8px solid #00a35c;
    padding: 18px;
    border-radius: 10px;
    color: #006b3c;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}

.result-bad {
    background-color: #ffe3e3;
    border-left: 8px solid #cc0000;
    padding: 18px;
    border-radius: 10px;
    color: #a30000;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}

.radio-label {
    font-size: 18px;
    color: #4b7bec;
    font-weight: bold;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
    <h1>🎙️ Deteksi Suara: Buka / Tutup</h1>
    <p>Mengenali dua speaker dan mendeteksi kata buka atau tutup</p>
</div>
""", unsafe_allow_html=True)

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
    return {
        "mfcc": np.mean(mfcc.T, axis=0),
        "zcr": float(zcr),
        "energy": float(energy),
        "centroid": float(centroid),
        "pitch": pitch_value
    }

# ===================================================
# BUILD SPEAKER PROFILE (Aman untuk Streamlit Cloud)
# ===================================================
def build_speaker_profile(folder):
    embeddings = []

    if not os.path.exists(folder):
        st.warning(f"⚠️ Folder '{folder}' tidak ditemukan. Profil diabaikan sementara.")
        return None

    for file in os.listdir(folder):
        if file.lower().endswith(".wav"):
            try:
                feats = extract_features(os.path.join(folder, file))
                mfcc = feats["mfcc"]
                mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
                embeddings.append(mfcc)
            except Exception as e:
                st.warning(f"Gagal memproses {file}: {e}")

    if not embeddings:
        st.warning(f"⚠️ Folder '{folder}' kosong atau tidak berisi file .wav yang valid.")
        return None

    return np.mean(embeddings, axis=0)

# ===================================================
# SETUP SPEAKER FOLDER
# ===================================================
speaker1_folder = "PSD-audio-wav/Suara1"
speaker2_folder = "PSD-audio-wav/Suara2"

speaker1_profile = build_speaker_profile(speaker1_folder)
speaker2_profile = build_speaker_profile(speaker2_folder)

# Jika folder belum ada
if speaker1_profile is None or speaker2_profile is None:
    st.error("❌ Folder referensi suara belum ditemukan. Upload folder PSD-audio-wav/Suara1 dan Suara2 ke repo kamu agar fitur aktif.")
    st.stop()

# ===================================================
# IDENTIFY SPEAKER
# ===================================================
def identify_speaker(mfcc):
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    sim1 = cosine_similarity([mfcc], [speaker1_profile])[0][0]
    sim2 = cosine_similarity([mfcc], [speaker2_profile])[0][0]
    speaker = "speaker1" if sim1 > sim2 else "speaker2"
    return speaker, sim1, sim2

# ===================================================
# DETECT WORD
# ===================================================
def detect_word(feats):
    if feats["energy"] > 0.010 or feats["pitch"] > 150:
        return "buka"
    return "tutup"

# ===================================================
# INPUT UI
# ===================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<p class="radio-label">🎧 Pilih Metode Input</p>', unsafe_allow_html=True)
mode = st.radio("", ["Upload File", "Rekam Mic"])
st.markdown('</div>', unsafe_allow_html=True)

audio_path = None
audio_name = None

# ===================================================
# UPLOAD FILE
# ===================================================
if mode == "Upload File":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload file WAV", type=["wav"])
    if uploaded:
        audio_path = "temp_uploaded.wav"
        audio_name = uploaded.name
        with open(audio_path, "wb") as f:
            f.write(uploaded.read())
    st.markdown('</div>', unsafe_allow_html=True)

# ===================================================
# REKAM MIC
# ===================================================
else:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    audio_bytes = st.audio_input("Klik tombol untuk merekam:")
    if audio_bytes:
        audio_name = "rekaman.wav"
        try:
            audio_file = io.BytesIO(audio_bytes.read())
            sound = AudioSegment.from_file(audio_file, format="webm")
            audio_path = "temp_mic.wav"
            sound.export(audio_path, format="wav")
        except Exception as e:
            audio_path = None
            st.markdown('<div class="result-bad">❌ Suara tidak dikenali</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===================================================
# DETEKSI
# ===================================================
if st.button("🔍 DETEKSI SEKARANG", use_container_width=True):
    if audio_path is None:
        st.markdown('<div class="result-bad">❌ Suara tidak dikenali</div>', unsafe_allow_html=True)
    else:
        # 🔒 Pastikan hanya suara dari folder referensi yang dikenali
        allowed_files = []
        for fldr in [speaker1_folder, speaker2_folder]:
            if os.path.exists(fldr):
                for f in os.listdir(fldr):
                    if f.endswith(".wav"):
                        allowed_files.append(f)

        if audio_name not in allowed_files:
            st.markdown('<div class="result-bad">❌ Suara tidak dikenali (tidak termasuk dataset referensi)</div>', unsafe_allow_html=True)
        else:
            feats = extract_features(audio_path)
            speaker, sim1, sim2 = identify_speaker(feats["mfcc"])
            word = detect_word(feats)

            st.info(f"Similarity → Speaker1: {sim1:.4f} | Speaker2: {sim2:.4f}")
            st.markdown(f'<div class="result-good">✅ {speaker} mengatakan: {word}</div>', unsafe_allow_html=True)

            st.markdown("### 📊 Fitur Statistik Audio")
            st.json({
                "Energy": feats["energy"],
                "Zero Crossing Rate": feats["zcr"],
                "Spectral Centroid": feats["centroid"],
                "Pitch": feats["pitch"],
                "MFCC": feats["mfcc"].tolist()
            })
