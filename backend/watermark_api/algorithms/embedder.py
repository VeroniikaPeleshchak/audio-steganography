import numpy as np
import pywt
import scipy.io.wavfile as wav

def embed_watermark(audio_path, output_path, watermark_text, alpha=0.005):
    sample_rate, audio_data = wav.read(audio_path)
    if len(audio_data.shape) > 1: audio_data = audio_data[:, 0]

    cA, cD = pywt.dwt(audio_data, 'haar')
    n = int(np.floor(np.sqrt(len(cA))))
    cA_truncated = cA[:n**2]
    matrix_cA = cA_truncated.reshape((n, n))

    U, S, Vt = np.linalg.svd(matrix_cA, full_matrices=False)

    if set(watermark_text).issubset({'0', '1'}) and len(watermark_text) % 8 == 0:
        watermark_bits = np.array([int(bit) for bit in watermark_text])
    else:
        byte_array = watermark_text.encode('utf-8')
        binary = ''.join(format(byte, '08b') for byte in byte_array)
        watermark_bits = np.array([int(bit) for bit in binary])

    if len(watermark_bits) > len(S):
        watermark_bits = watermark_bits[:len(S)]

    W = np.resize(watermark_bits, len(S))
    W_bipolar = np.where(W == 1, 1, -1)

    S_new = S + alpha * W_bipolar * np.max(S)
    matrix_cA_new = U @ np.diag(S_new) @ Vt

    cA_new = matrix_cA_new.flatten()
    cA_modified = np.concatenate((cA_new, cA[n**2:]))

    reconstructed_audio = pywt.idwt(cA_modified, cD, 'haar')

    reconstructed_audio = np.clip(reconstructed_audio, -32768, 32767)
    reconstructed_audio = np.int16(reconstructed_audio)

    wav.write(output_path, sample_rate, reconstructed_audio)
    return True