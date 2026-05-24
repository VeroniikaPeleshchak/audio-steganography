from reedsolo import RSCodec

rs = RSCodec(10)

def encode_text_with_rs(text: str) -> str:
    text_bytes = text.encode('utf-8')
    encoded_bytes = rs.encode(text_bytes)
    return ''.join(format(byte, '08b') for byte in encoded_bytes)

def decode_text_with_rs(extracted_bits: str) -> tuple[str, bool]:
    valid_length = len(extracted_bits) - (len(extracted_bits) % 8)
    extracted_bits = extracted_bits[:valid_length]

    try:
        byte_array = bytearray(int(extracted_bits[i:i+8], 2) for i in range(0, len(extracted_bits), 8))
        decoded_bytes = rs.decode(byte_array)[0]
        return decoded_bytes.decode('utf-8'), True
    except Exception:
        return "ПОМИЛКА: Рівень шуму перевищив можливості Ріда-Соломона", False