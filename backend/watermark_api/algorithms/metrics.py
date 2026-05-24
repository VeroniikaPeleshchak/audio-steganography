import numpy as np
import difflib
import math

def calculate_snr(original_signal, watermarked_signal):
    if len(original_signal.shape) > 1: original_signal = original_signal[:, 0]
    if len(watermarked_signal.shape) > 1: watermarked_signal = watermarked_signal[:, 0]
    min_len = min(len(original_signal), len(watermarked_signal))
    orig = original_signal[:min_len].astype(np.float64)
    wat = watermarked_signal[:min_len].astype(np.float64)
    sig_pwr = np.sum(orig ** 2)
    noise_pwr = np.sum((orig - wat) ** 2)
    return round(10 * np.log10(sig_pwr / noise_pwr), 2) if noise_pwr > 0 else 100.0

def calculate_match_percentage(orig_text, ext_text):
    if not orig_text or not ext_text: return 0.0
    # Очищаємо текст від невидимих символів
    orig_clean = str(orig_text).strip().replace('\x00', '')
    ext_clean = str(ext_text).strip().replace('\x00', '')
    if not ext_clean or ext_clean.startswith('❌'): return 0.0
    return round(difflib.SequenceMatcher(None, orig_clean, ext_clean).ratio() * 100, 2)

def calculate_ber(orig_text, ext_text):
    return round(100.0 - calculate_match_percentage(orig_text, ext_text), 2)

def calculate_psnr(original_signal, watermarked_signal):
    if len(original_signal.shape) > 1: original_signal = original_signal[:, 0]
    if len(watermarked_signal.shape) > 1: watermarked_signal = watermarked_signal[:, 0]
    min_len = min(len(original_signal), len(watermarked_signal))
    orig = original_signal[:min_len].astype(np.float64)
    wat = watermarked_signal[:min_len].astype(np.float64)
    mse = np.mean((orig - wat) ** 2)
    return round(20 * math.log10(32767.0 / math.sqrt(mse)), 2) if mse > 0 else 100.0