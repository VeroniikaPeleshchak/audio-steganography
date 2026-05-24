import numpy as np
import scipy.signal as signal
import io
import scipy.io.wavfile as wav
from pydub import AudioSegment

def add_white_noise(audio_data, snr_target_db=20):
    audio_float = audio_data.astype(np.float64)
    
    signal_power = np.mean(audio_float ** 2)
    signal_power_db = 10 * np.log10(signal_power if signal_power > 0 else 1e-10)

    noise_power_db = signal_power_db - snr_target_db
    noise_power = 10 ** (noise_power_db / 10)
    
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio_float))
    
    attacked_audio = audio_float + noise
    attacked_audio = np.clip(attacked_audio, -32768, 32767)
    
    return attacked_audio.astype(np.int16)


def apply_lowpass_filter(audio_data, sample_rate, cutoff_freq=3000):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    
    filtered_audio = signal.filtfilt(b, a, audio_data)
    
    return np.clip(filtered_audio, -32768, 32767).astype(np.int16)


def crop_audio(audio_data, crop_percent=15):
    if crop_percent >= 100 or crop_percent <= 0:
        return audio_data
        
    keep_length = int(len(audio_data) * (1 - crop_percent / 100.0))
    
    attacked_audio = np.zeros_like(audio_data)
    attacked_audio[:keep_length] = audio_data[:keep_length]
    
    return attacked_audio

def scale_amplitude(audio_data, factor=0.5):
    attacked_audio = audio_data.astype(np.float64) * factor
    return np.clip(attacked_audio, -32768, 32767).astype(np.int16)

def apply_mp3_compression(audio_data, sample_rate, bitrate="128k"):
    wav_io = io.BytesIO()
    wav.write(wav_io, sample_rate, audio_data.astype(np.int16))
    wav_io.seek(0)
    
    audio_seg = AudioSegment.from_file(wav_io, format="wav")
    mp3_io = io.BytesIO()
    audio_seg.export(mp3_io, format="mp3", bitrate=bitrate)
    mp3_io.seek(0)
    
    audio_seg_mp3 = AudioSegment.from_file(mp3_io, format="mp3")
    final_wav_io = io.BytesIO()
    audio_seg_mp3.export(final_wav_io, format="wav")
    final_wav_io.seek(0)
    
    sr, compressed_data = wav.read(final_wav_io)
    
    if len(compressed_data) > len(audio_data):
        compressed_data = compressed_data[:len(audio_data)]
    elif len(compressed_data) < len(audio_data):
        compressed_data = np.pad(compressed_data, (0, len(audio_data) - len(compressed_data)), 'constant')
        
    return compressed_data