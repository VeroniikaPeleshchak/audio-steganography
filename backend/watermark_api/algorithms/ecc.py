import numpy as np
import random

def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_text(binary_str):
    chars = [binary_str[i:i+8] for i in range(0, len(binary_str), 8)]
    res = ""
    for c in chars:
        if len(c) == 8:
            try:
                val = int(c, 2)
                if 32 <= val <= 1111: res += chr(val)
                else: res += '?'
            except: res += '?'
    return res

def hamming_encode(text):
    data_bits = text_to_binary(text)
    encoded = []
    while len(data_bits) % 4 != 0: data_bits += '0'
    
    for i in range(0, len(data_bits), 4):
        d = [int(x) for x in data_bits[i:i+4]]
        p1, p2, p3 = (d[0]+d[1]+d[3])%2, (d[0]+d[2]+d[3])%2, (d[1]+d[2]+d[3])%2
        encoded.extend([p1, p2, d[0], p3, d[1], d[2], d[3]])
    
    idx = list(range(len(encoded)))
    random.seed(42)
    random.shuffle(idx)
    shuffled = [0]*len(encoded)
    for i, v in enumerate(idx): shuffled[v] = encoded[i]
    return ''.join(map(str, shuffled))

def hamming_decode(bits_str):
    if not bits_str or len(bits_str) < 7: return ""
    bits = [int(b) for b in bits_str if b in '01']
    
    idx = list(range(len(bits)))
    random.seed(42)
    random.shuffle(idx)
    orig_order = [0]*len(bits)
    for i, v in enumerate(idx): orig_order[i] = bits[v]
    
    decoded_bits = ""
    for i in range(0, len(orig_order)-6, 7):
        c = orig_order[i:i+7]
        s1, s2, s3 = (c[0]+c[2]+c[4]+c[6])%2, (c[1]+c[2]+c[5]+c[6])%2, (c[3]+c[4]+c[5]+c[6])%2
        syn = s1 + s2*2 + s3*4
        if syn: c[syn-1] ^= 1 
        decoded_bits += f"{c[2]}{c[4]}{c[5]}{c[6]}"
    
    return binary_to_text(decoded_bits).strip('\x00')