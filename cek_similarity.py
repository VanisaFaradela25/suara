import librosa
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Fungsi ekstraksi fitur
# -------------------------------
def extract_features(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc.T, axis=0)

# -------------------------------
# Buat profil speaker
# -------------------------------
def build_profile(folder):
    emb = []
    for f in os.listdir(folder):
        if f.endswith(".wav"):
            mf = extract_features(os.path.join(folder, f))
            mf = (mf - np.mean(mf)) / (np.std(mf) + 1e-9)
            emb.append(mf)
    return np.mean(emb, axis=0)

# Folder Anda
S1 = "PSD-audio-wav/Suara1"
S2 = "PSD-audio-wav/Suara2"

speaker1 = build_profile(S1)
speaker2 = build_profile(S2)

# -------------------------------
# Cek similarity tiap file
# -------------------------------
def cek_folder(folder, label):
    print(f"\n=== {label} ===")
    for f in os.listdir(folder):
        if f.endswith(".wav"):
            path = os.path.join(folder, f)
            mf = extract_features(path)
            mf = (mf - np.mean(mf)) / (np.std(mf) + 1e-9)

            sim1 = cosine_similarity([mf], [speaker1])[0][0]
            sim2 = cosine_similarity([mf], [speaker2])[0][0]

            print(f"{f}: sim1 = {sim1:.4f}, sim2 = {sim2:.4f}")

# Cek similarity
cek_folder(S1, "Suara 1 → similarity")
cek_folder(S2, "Suara 2 → similarity")
