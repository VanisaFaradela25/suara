import os
from pydub import AudioSegment

# Lokasi ffmpeg & ffprobe (gunakan punya kamu)
AudioSegment.converter = "ffmpeg.exe"
AudioSegment.ffprobe   = "ffprobe.exe"

src_root = r"PSD-audio"
dst_root = r"PSD-audio-wav"

folders = ["Suara1", "Suara2"]

for f in folders:
    src_folder = os.path.join(src_root, f)
    dst_folder = os.path.join(dst_root, f)

    os.makedirs(dst_folder, exist_ok=True)

    for file in os.listdir(src_folder):
        if file.endswith((".m4a", ".mp3")):
            src = os.path.join(src_folder, file)
            dst = os.path.join(dst_folder, file.rsplit(".", 1)[0] + ".wav")
            print("Converting:", file)
            AudioSegment.from_file(src).export(dst, format="wav")

print("SELESAI.")
