import numpy as np
import scipy.io.wavfile as wav

sample_rate = 44100

# Багатий синтетичний ПЕД (Rich Pad)

t_pad = np.linspace(0, 4.0, int(sample_rate * 4.0), False)
pad_signal = (
    np.sin(2 * np.pi * 150 * t_pad) + 
    np.sin(2 * np.pi * 300 * t_pad) + 
    np.sin(2 * np.pi * 600 * t_pad) + 
    0.1 * np.random.normal(0, 1, len(t_pad))
) / 3.1
pad_envelope = np.exp(-0.2 * t_pad)
pad_signal = pad_signal * pad_envelope

wav.write("test_rich_pad.wav", sample_rate, np.int16(pad_signal * 32767))


# Чистий Саб-бас 
t_bass = np.linspace(0, 4.0, int(sample_rate * 4.0), False)
bass_signal = np.sin(2 * np.pi * 50 * t_bass)

fade = np.linspace(0, 1, 2000)
bass_signal[:2000] *= fade
bass_signal[-2000:] *= fade[::-1]

wav.write("test_sub_bass.wav", sample_rate, np.int16(bass_signal * 32767))


# Імпульси з тишею (Sparse Staccato)
t_total = 4.0
sparse_signal = np.zeros(int(sample_rate * t_total))

beep_dur = 0.2
t_beep = np.linspace(0, beep_dur, int(sample_rate * beep_dur), False)
beep = np.sin(2 * np.pi * 1000 * t_beep) * np.exp(-8 * t_beep)

intervals = [0.5, 1.5, 2.5, 3.5]
for start_t in intervals:
    start_idx = int(start_t * sample_rate)
    end_idx = start_idx + len(beep)
    sparse_signal[start_idx:end_idx] = beep

wav.write("test_sparse.wav", sample_rate, np.int16(sparse_signal * 32767))
