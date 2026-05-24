import os
import numpy as np
from scipy.io import wavfile

def generate_ecc_test_audio(filename="ecc_test.wav", duration=3.0):
    sample_rate = 44100  
    frequency = 440.0   
    
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    clean_signal = 32700 * np.sin(2 * np.pi * frequency * t)
    
    noise_amplitude = 2 
    micro_noise = np.random.uniform(-noise_amplitude, noise_amplitude, len(t))
    
    noisy_signal = clean_signal + micro_noise
    
    final_signal = np.int16(np.clip(noisy_signal, -32768, 32767))
    
    output_dir = "audio"
    
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, filename)
    
    wavfile.write(file_path, sample_rate, final_signal)

generate_ecc_test_audio()