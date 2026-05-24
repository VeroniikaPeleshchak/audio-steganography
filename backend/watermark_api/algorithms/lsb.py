import numpy as np

def text_to_bits(text):
    bits = []
    for char in text:
        binval = bin(ord(char))[2:].rjust(8, '0')
        bits.extend([int(b) for b in binval])
    return bits

def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:
            char_code = int(''.join(map(str, byte)), 2)
            if 32 <= char_code <= 126 or 1040 <= char_code <= 1111 or char_code == 61: # Додано 61 для '=' (Base64)
                chars.append(chr(char_code))
            else:
                chars.append('?')
    return ''.join(chars)

def embed_lsb(audio_data, watermark_text):
    if len(audio_data.shape) > 1: audio_data = audio_data[:, 0] 
        
    watermark_bits = text_to_bits(watermark_text)
    
    watermark_bits.extend([0, 0, 0, 0, 0, 0, 0, 0]) 
    watermark_bits.extend([1] * 16) 
    
    if len(watermark_bits) > len(audio_data):
        raise ValueError("Текст занадто довгий для цього аудіофайлу.")
        
    watermarked_audio = np.copy(audio_data)
    
    for i in range(len(watermark_bits)):
        watermarked_audio[i] = (watermarked_audio[i] & ~1) | watermark_bits[i]
        
    return watermarked_audio

def extract_lsb(audio_data):
    if len(audio_data.shape) > 1: audio_data = audio_data[:, 0] 
        
    extracted_bits = []
    consecutive_ones = 0
    
    for i in range(len(audio_data)):
        bit = audio_data[i] & 1
        extracted_bits.append(bit)
        
        if bit == 1:
            consecutive_ones += 1
        else:
            consecutive_ones = 0
            
        if consecutive_ones == 16:
            extracted_bits = extracted_bits[:-16] 
            break
            
    return bits_to_text(extracted_bits)