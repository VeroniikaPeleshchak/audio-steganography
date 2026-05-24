import os
import time
import json
import shutil
import uuid
import scipy.io.wavfile as wav 
import numpy as np
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.conf import settings
from django.core.files.storage import FileSystemStorage 

from .models import AudioTrack, ExpertiseResult
from .algorithms.embedder import embed_watermark
from .algorithms.extractor import extract_watermark     
from .algorithms.metrics import calculate_match_percentage, calculate_snr, calculate_psnr, calculate_ber 
from .algorithms.lsb import embed_lsb, extract_lsb 
from .algorithms.attacks import add_white_noise, apply_lowpass_filter, crop_audio, scale_amplitude, apply_mp3_compression
from .algorithms.ecc import hamming_encode, hamming_decode 
from .algorithms.ecc_rs import encode_text_with_rs, decode_text_with_rs 

@api_view(['POST'])
def protect_audio(request):
    audio_file = request.FILES.get('audio')
    watermark_text = request.data.get('watermark_data', 'secretText')
    algorithm = request.data.get('algorithm', 'dwt_svd')
    use_ecc = request.data.get('use_ecc', 'false').lower() == 'true' 

    if not audio_file:
        return Response({"error": "Файл не знайдено"}, status=400)

    start_time = time.time()


    if use_ecc:
        if algorithm == 'lsb':
            watermark_text = hamming_encode(watermark_text) 
        else:
            watermark_text = encode_text_with_rs(watermark_text) 

    track = AudioTrack.objects.create(
        original_audio=audio_file,
        watermark_data=watermark_text,
        algorithm=f"{algorithm}_ecc" if use_ecc else algorithm, 
        is_ecc=use_ecc
    )

    input_path = track.original_audio.path
    output_dir = os.path.join(settings.MEDIA_ROOT, 'uploads/watermarked')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"protected_{track.id}.wav")

    sample_rate, audio_data = wav.read(input_path)
    capacity = round((len(watermark_text.encode('utf-8')) / audio_data.nbytes) * 100, 5)

    if algorithm == 'lsb':
        try:
            protected_data = embed_lsb(audio_data, watermark_text)
            wav.write(output_path, sample_rate, protected_data.astype(np.int16))
            success = True
        except Exception as e:
            success = False
    else:
        success = embed_watermark(input_path, output_path, watermark_text)

    execution_time = round(time.time() - start_time, 4)

    if success:
        track.watermarked_audio.name = f"uploads/watermarked/protected_{track.id}.wav"

        try:
            _, orig_data = wav.read(input_path)
            _, wat_data = wav.read(output_path)
            track.snr = calculate_snr(orig_data, wat_data)
            track.psnr = calculate_psnr(orig_data, wat_data)
        except Exception as e:
            print("Помилка розрахунку метрик:", e)

        track.save()
        return Response({
            "message": "Водяний знак успішно вшито",
            "track_id": track.id,
            "protected_file": track.watermarked_audio.name,
            "execution_time": execution_time,
            "capacity": f"{capacity}%",
            "algorithm": track.algorithm
        })
    return Response({"error": "Помилка обробки"}, status=500)
    

@api_view(['POST'])
def check_audio(request):
    track_id = request.data.get('track_id')
    suspect_file = request.FILES.get('suspect_audio')
    attacks_list = json.loads(request.data.get('attacks', '[]'))
    manual_algorithm = request.data.get('manual_algorithm', 'dwt_svd')
    
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    suspect_path = ""
    original_path = None
    algorithm = manual_algorithm
    original_text = ""
    is_temp_file = False

    if track_id:
        try:
            track = AudioTrack.objects.get(id=track_id)
            original_path = track.original_audio.path
            original_text = track.watermark_data
            algorithm = track.algorithm
            
            if not suspect_file:
                temp_filename = f"temp_{uuid.uuid4().hex}.wav"
                suspect_path = os.path.join(temp_dir, temp_filename)
                shutil.copy2(track.watermarked_audio.path, suspect_path)
                is_temp_file = True
            else:
                fs = FileSystemStorage(location=temp_dir)
                filename = fs.save(suspect_file.name, suspect_file)
                suspect_path = fs.path(filename)
                is_temp_file = True
        except AudioTrack.DoesNotExist:
            return Response({"error": "Трек не знайдено."}, status=404)
    else:
        if not suspect_file:
            return Response({"error": "Вкажіть ID або файл!"}, status=400)
        fs = FileSystemStorage(location=temp_dir)
        filename = fs.save(suspect_file.name, suspect_file)
        suspect_path = fs.path(filename)
        is_temp_file = True

    start_time = time.time()
    sample_rate, suspect_data = wav.read(suspect_path)
    
    attack_names_applied = []
    for att in attacks_list:
        atype = att.get('type')
        raw_param = att.get('param', 0)
        
        if atype == 'mp3':
            aparam = raw_param
        else:
            aparam = float(raw_param)

        if atype == 'noise': 
            suspect_data = add_white_noise(suspect_data, aparam)
            attack_names_applied.append('Шум')
        elif atype == 'lowpass': 
            suspect_data = apply_lowpass_filter(suspect_data, sample_rate, aparam)
            attack_names_applied.append('Фільтр')
        elif atype == 'crop': 
            suspect_data = crop_audio(suspect_data, aparam)
            attack_names_applied.append('Обрізання')
        elif atype == 'volume': 
            suspect_data = scale_amplitude(suspect_data, aparam)
            attack_names_applied.append('Зміна гучності')
        elif atype == 'mp3':
            suspect_data = apply_mp3_compression(suspect_data, sample_rate, aparam)
            attack_names_applied.append(f'MP3 {aparam}')

    if attack_names_applied:
        wav.write(suspect_path, sample_rate, suspect_data)
        attack_str = " + ".join(attack_names_applied)
    else:
        attack_str = "Без атак"

    extracted_text = ""
    base_algo = 'lsb' if 'lsb' in algorithm else 'dwt_svd'
    
    if base_algo == 'lsb':
        try:
            _, final_data = wav.read(suspect_path)
            raw_extracted = extract_lsb(final_data)
            extracted_text = hamming_decode(raw_extracted) if '_ecc' in algorithm else raw_extracted
        except Exception:
            extracted_text = "Помилка читання"
    else:
        if not original_path:
            extracted_text = "Неможливо без оригіналу"
        else:
            raw_extracted = extract_watermark(original_path, suspect_path, original_text)
            if '_ecc' in algorithm:
                extracted_text, _ = decode_text_with_rs(raw_extracted)
            else:
                extracted_text = raw_extracted

    execution_time = round(time.time() - start_time, 4)

    if '_ecc' in algorithm:
        if 'lsb' in algorithm:
            display_original = hamming_decode(original_text)
        else:
            display_original, _ = decode_text_with_rs(original_text)
    else:
        display_original = original_text

    res = {
        "message": "Експертиза завершена",
        "extracted_watermark": extracted_text,
        "execution_time": execution_time,
        "algorithm": algorithm,
        "attack_applied": attack_str,
        "ecc_used": '_ecc' in algorithm
    }

    if original_path:
        res["original_watermark"] = display_original
        match_percent = calculate_match_percentage(display_original, extracted_text)
        
        if not str(extracted_text).strip():
            extracted_text = "[Порожньо]"

        res["match_percentage"] = match_percent
        res["ber"] = calculate_ber(display_original, extracted_text)
        try:
            _, orig_data = wav.read(original_path)
            res["snr"] = calculate_snr(orig_data, suspect_data)
            res["psnr"] = calculate_psnr(orig_data, suspect_data)
        except Exception:
            res["snr"], res["psnr"] = "N/A", "N/A"

        if track_id:
            try:
                ExpertiseResult.objects.create(
                    track_id=track_id,
                    attack_type=attack_str,
                    attack_params=str(attacks_list),
                    extracted_text=extracted_text,
                    ber=res["ber"],
                    ncc=match_percent / 100.0
                )
            except Exception as e:
                print("Помилка збереження результату:", e)

    if is_temp_file and os.path.exists(suspect_path):
        os.remove(suspect_path)

    return Response(res)


@api_view(['POST'])
def run_benchmark(request):
    track_id = request.data.get('track_id')
    try:
        track = AudioTrack.objects.get(id=track_id)
    except AudioTrack.DoesNotExist:
        return Response({"error": "Трек не знайдено"}, status=404)

    original_path = track.original_audio.path
    watermarked_path = track.watermarked_audio.path
    original_text = track.watermark_data
    algorithm = track.algorithm
    
    _, orig_data = wav.read(original_path)
    sample_rate, protected_data = wav.read(watermarked_path)

    test_suite = [
        {"name": "Еталон (Без атак)", "type": "none", "param": 0},
        {"name": "Шум (AWGN) 30 dB", "type": "noise", "param": 30},
        {"name": "Шум (AWGN) 15 dB", "type": "noise", "param": 15},
        {"name": "Фільтр 4000 Hz", "type": "lowpass", "param": 4000},
        {"name": "Обрізання 10%", "type": "crop", "param": 10},
        {"name": "MP3 Стиснення 128kbps", "type": "mp3", "param": "128k"},
        {"name": "MP3 Стиснення 64kbps", "type": "mp3", "param": "64k"},
    ]

    results = []
    
    for test in test_suite:
        test_data = protected_data.copy()
        
        if test['type'] == 'noise': test_data = add_white_noise(test_data, test['param'])
        elif test['type'] == 'lowpass': test_data = apply_lowpass_filter(test_data, sample_rate, test['param'])
        elif test['type'] == 'crop': test_data = crop_audio(test_data, test['param'])
        elif test['type'] == 'mp3': test_data = apply_mp3_compression(test_data, sample_rate, test['param'])

        temp_path = os.path.join(settings.MEDIA_ROOT, 'temp', f"bench_{uuid.uuid4().hex}.wav")
        wav.write(temp_path, sample_rate, test_data)

        extracted_text = ""
        base_algo = 'lsb' if 'lsb' in algorithm else 'dwt_svd'
        
        if base_algo == 'lsb':
            _, final_data = wav.read(temp_path)
            raw_ex = extract_lsb(final_data)
            extracted_text = hamming_decode(raw_ex) if '_ecc' in algorithm else raw_ex
        else:
            raw_ex = extract_watermark(original_path, temp_path, original_text)
            if '_ecc' in algorithm:
                extracted_text, _ = decode_text_with_rs(raw_ex)
            else:
                extracted_text = raw_ex

        if '_ecc' in algorithm:
            if 'lsb' in algorithm:
                display_original = hamming_decode(original_text)
            else:
                display_original, _ = decode_text_with_rs(original_text)
        else:
            display_original = original_text

        match_percent = calculate_match_percentage(display_original, extracted_text)
        ncc_val = round(match_percent / 100.0, 4)
        ber_val = calculate_ber(display_original, extracted_text)
        snr_val = calculate_snr(orig_data, test_data)
        psnr_val = calculate_psnr(orig_data, test_data)
        
        if not str(extracted_text).strip():
            extracted_text = "[Порожньо]"
        
        results.append({
            "Експеримент": test['name'],
            "NCC": ncc_val,
            "BER (%)": ber_val,
            "SNR (dB)": snr_val,
            "PSNR (dB)": psnr_val,
            "Витягнутий текст": extracted_text
        })

        try:
            ExpertiseResult.objects.create(
                track=track,
                attack_type=test['name'],
                attack_params=str(test['param']),
                extracted_text=extracted_text,
                ber=ber_val,
                ncc=ncc_val,
                snr=snr_val,
                psnr=psnr_val
            )
        except Exception as e:
            print("Помилка збереження бенчмарку:", e)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return Response({"benchmark_results": results})


@api_view(['GET'])
def test_connection(request):
    return Response({"message": "Зв'язок встановлено! Бекенд готовий приймати аудіо."})