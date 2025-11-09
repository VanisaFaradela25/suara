import os
from pydub import AudioSegment

# Panggil FFmpeg
AudioSegment.converter = r"C:\Users\vanisa\OneDrive\Documents\SEMESTER 5\Proyek Sains Data\proyek\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"

def convert_folder_to_wav(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        
        if not os.path.isfile(filepath):
            continue
        
        name, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext not in [".m4a", ".mp3"]:
            continue

        audio = AudioSegment.from_file(filepath)
        audio.export(os.path.join(output_folder, f"{name}.wav"), format="wav")
        print(f"Converted: {filename}")

# Panggil folder kamu
convert_folder_to_wav(
    r"C:\Users\vanisa\OneDrive\Documents\SEMESTER 5\Proyek Sains Data\proyek\PSD-audio\Suara1",
    r"C:\Users\vanisa\OneDrive\Documents\SEMESTER 5\Proyek Sains Data\proyek\PSD-audio\Suara1_wav"
)

convert_folder_to_wav(
    r"C:\Users\vanisa\OneDrive\Documents\SEMESTER 5\Proyek Sains Data\proyek\PSD-audio\Suara2",
    r"C:\Users\vanisa\OneDrive\Documents\SEMESTER 5\Proyek Sains Data\proyek\PSD-audio\Suara2_wav"
)
