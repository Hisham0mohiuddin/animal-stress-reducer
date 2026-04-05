import librosa
import librosa.display
import numpy as np
import os
import soundfile as sf
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

def apply_bird_filter(audio,sr):
    nyquist = 0.5 * sr
    low = 1500 / nyquist
    high = 10000 / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_audio = lfilter(b, a, audio)

    yt, index = librosa.effects.trim(filtered_audio, top_db=20)

    return yt 

def standard_equal_l(file_path, file_length = 10 , sr = 22050):
    y, sr = librosa.load(file_path, sr=sr)
    y_filtered = apply_bird_filter(y, sr)    
    target_no_sampels = sr * file_length
    current_no_samples = len(y_filtered)
    if(current_no_samples>=target_no_sampels):
        y_filtered_final = y_filtered[:target_no_sampels]
    else:
        padding_needed = target_no_sampels-current_no_samples
        y_filtered_final = np.pad(y_filtered, (padding_needed // 2, padding_needed - padding_needed // 2), mode='constant')
    
    mel_spec = librosa.feature.melspectrogram(y=y_filtered_final, sr=sr, n_mels=128)

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    #normalizing here itslef to save work later , maxmin as domain is known 
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    #to return 3d image for resnet to use, to emualate rgb values of same rgb
    mel_spec_reshaped_resnet = np.repeat(mel_spec_norm[:,:,np.newaxis], 3 , axis=-1)
    return mel_spec_reshaped_resnet,sr


if __name__ =="__main__":
    home = os.getcwd()
    file_path = os.path.join(home,"data/call/832368.mp3")
    y_filtered,sr = standard_equal_l(file_path)
    print(y_filtered.shape)
    plt.figure(figsize=(12,5))
    plt.imshow(y_filtered[:,:,0])
    plt.show()
