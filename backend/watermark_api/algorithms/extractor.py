import numpy as np
import pywt
import scipy.io.wavfile as wav

def extract_watermark(original_audio_path, watermarked_audio_path, original_text, alpha=0.005):
    sr_orig, orig_data = wav.read(original_audio_path)
    sr_wat, wat_data = wav.read(watermarked_audio_path)

    if len(orig_data.shape) > 1: orig_data = orig_data[:, 0]
    if len(wat_data.shape) > 1: wat_data = wat_data[:, 0]

    cA_orig, _ = pywt.dwt(orig_data, 'haar')
    cA_wat, _ = pywt.dwt(wat_data, 'haar')

    n = int(np.floor(np.sqrt(len(cA_orig))))

    matrix_cA_orig = cA_orig[:n**2].reshape((n, n))
    U, S_orig, Vt = np.linalg.svd(matrix_cA_orig, full_matrices=False)

    matrix_cA_wat = cA_wat[:n**2].reshape((n, n))

    S_wat_matrix = U.T @ matrix_cA_wat @ Vt.T
    S_wat = np.diag(S_wat_matrix)

    W_extracted = (S_wat - S_orig) / (alpha * np.max(S_orig))
    W_binary = np.where(W_extracted > 0, 1, 0)

    if set(original_text).issubset({'0', '1'}) and len(original_text) % 8 == 0:
        target_bits_count = len(original_text)
        is_raw_bits = True
    else:
        target_bits_count = len(original_text.encode('utf-8')) * 8
        is_raw_bits = False

    target_bits_count = min(target_bits_count, len(S_orig))
    extracted_bits = W_binary[:target_bits_count]

    bit_string = ''.join(str(int(b)) for b in extracted_bits)

    if is_raw_bits:
        return bit_string 
    else:
        bytes_list = [int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8) if len(bit_string[i:i+8]) == 8]
        extracted_text = bytes(bytes_list).decode('utf-8', errors='replace')
        return ''.join(c if c.isprintable() else '?' for c in extracted_text)