import librosa
import numpy as np
import os
import soundfile as sf
from scipy.signal import butter, lfilter

def apply_bird_filter(audio,sr):
    nyquist = 0.5 * sr
    low = 1500 / nyquist
    high = 10000 / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_audio = lfilter(b, a, audio)

    yt, index = librosa.effects.trim(filtered_audio, top_db=20)

    return yt 

def standard_equal_l(file_path, file_length = 10 , sr = 22050):
    y_filtered = apply_bird_filter(*librosa.load(file_path))
    target_no_sampels = sr * file_length
    current_no_samples = len(y_filtered)
    if(current_no_samples>=target_no_sampels):
        y_filtered_final = y_filtered[:target_no_sampels]
    else:
        padding_needed = target_no_sampels-current_no_samples
        y_filtered_final = np.pad(y_filtered, (padding_needed // 2, padding_needed - padding_needed // 2), mode='constant')
    return y_filtered_final,sr

home = os.getcwd()
file_path = os.path.join(home,"data/audio/122771.mp3")
y_filtered,sr = standard_equal_l(file_path)
print(y_filtered)
sf.write('cleaned_great_tit.wav', y_filtered, sr)
