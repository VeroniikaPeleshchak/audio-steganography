import streamlit as st
import requests
import plotly.graph_objects as go
import numpy as np
from streamlit_option_menu import option_menu
import io
import json
import datetime
import pandas as pd
import scipy.signal as signal
import scipy.io.wavfile as wav 
from pydub import AudioSegment

st.set_page_config(page_title="Audio Watermark Studio", layout="wide", page_icon="🎧")

st.markdown("""
<style>
    h1, h2, h3 { color: #04AA6D !important; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: #04AA6D !important; }
    .nav-link-icon { color: inherit !important; }
    .nav-link .nav-link-icon { color: #04AA6D !important; }
    .stButton>button { border-radius: 8px; font-weight: bold; }
    .nav-link { color: var(--text-color) !important; }
    .nav-link:hover { background-color: var(--secondary-background-color) !important; }
    .nav-link.active { background-color: #3b3b3b !important; color: white !important; }
    .nav-link.active .nav-link-icon { color: white !important; }
</style>
""", unsafe_allow_html=True)

if "shopping_cart" not in st.session_state:
    st.session_state.shopping_cart = []

with st.sidebar:
    st.markdown("### Панель керування")
    page = option_menu(
        menu_title=None,
        options=["Вбудовування", "Експертиза", "Авто-Бенчмарк", "Кошик"],
        icons=["shield-lock", "search", "bar-chart-fill", "cart3"],
        menu_icon="cast",
        default_index=0,
        styles={"container": {"padding": "0!important", "background-color": "transparent"}, "nav-link": {"font-size": "16px"}}
    )
    st.markdown("---")
    st.write(f"Файлів у кошику: **{len(st.session_state.shopping_cart)}**")


def plot_real_audio(file_bytes, view_type="Осцилограма"):
    try:
        sr, data_audio = wav.read(io.BytesIO(file_bytes))
        if len(data_audio.shape) > 1:
            data_audio = data_audio[:, 0] 
        
        if view_type == "Осцилограма":
            step = max(1, len(data_audio) // 3000) 
            y = data_audio[::step]
            t = np.linspace(0, len(data_audio)/sr, len(y))
            fig = go.Figure(data=go.Scatter(x=t, y=y, mode='lines', line=dict(color='#04AA6D', width=1)))
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=200, xaxis=dict(visible=False), yaxis=dict(visible=False))
            return fig
        else:
            max_len = sr * 5
            data_audio = data_audio[:max_len]
            f, t, Sxx = signal.spectrogram(data_audio, sr)
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            fig = go.Figure(data=go.Heatmap(z=Sxx_db, x=t, y=f, colorscale='Viridis'))
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=250, xaxis_title="Час (с)", yaxis_title="Частота (Hz)")
            return fig
    except Exception as e:
        return None

def generate_report_content(data, track_id):
    ecc_type = "Хеммінга" if 'lsb' in data.get('algorithm', '') else "Ріда-Соломона"
    
    report = f"===============================================\n"
    report += f"     ЗВІТ ЕКСПЕРТИЗИ СТЕГОКОНТЕЙНЕРА\n"
    report += f"===============================================\n"
    report += f"Дата генерації: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"ID Треку: {track_id if track_id else 'Сторонній файл'}\n"
    report += f"Алгоритм вбудовування: {data.get('algorithm')}\n"
    report += f"Використання ЕСС ({ecc_type}): {'ТАК' if data.get('ecc_used') else 'НІ'}\n"
    report += f"Застосовані кібератаки: {data.get('attack_applied')}\n\n"
    report += f"--- РЕЗУЛЬТАТИ ВИТЯГНЕННЯ ДАНИХ ---\n"
    report += f"Оригінальний знак: {data.get('original_watermark', 'N/A')}\n"
    report += f"Витягнутий знак:   {data.get('extracted_watermark', 'N/A')}\n"
    report += f"Точність збігу (NCC): {data.get('match_percentage', 'N/A')}%\n"
    report += f"Помилка бітів (BER):  {data.get('ber', 'N/A')}%\n\n"
    report += f"--- АКУСТИЧНА ДЕГРАДАЦІЯ (ФІЗИКА ЗВУКУ) ---\n"
    report += f"Відношення Сигнал/Шум (SNR): {data.get('snr', 'N/A')} dB\n"
    report += f"Пікова якість (PSNR):        {data.get('psnr', 'N/A')} dB\n"
    report += f"Час роботи алгоритму:        {data.get('execution_time', 'N/A')} сек\n"
    report += f"===============================================\n"
    return report


if page == "Вбудовування":
    st.title("Вбудовування водяного знаку")
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown("#### 1. Налаштування захисту")
        uploaded_file = st.file_uploader("Оберіть .wav або .mp3 файл", type=["wav", "mp3"])
        watermark_text = st.text_input("Секретний текст:", "secretText")
        algo_choice = st.radio("Оберіть алгоритм:", ["DWT+SVD", "LSB"])
        
        use_ecc = st.toggle("Увімкнути завадостійке кодування (ECC: Хеммінг або Рід-Соломон)", value=True, help="Автоматично виправляє помилки від шуму.")
        submit_btn = st.button("🚀 Захистити файл", type="primary", use_container_width=True)

    with col2:
        st.markdown("#### 2. Аналіз сигналу")
        if uploaded_file is not None:
            view_type = st.radio("Вид графіка:", ["Осцилограма", "Спектрограма (2D Частоти)"], horizontal=True)
            file_bytes = uploaded_file.getvalue()
            
            if uploaded_file.name.endswith('.mp3'):
                audio_seg = AudioSegment.from_file(io.BytesIO(file_bytes), format="mp3")
                wav_io = io.BytesIO()
                audio_seg.export(wav_io, format="wav")
                file_bytes = wav_io.getvalue()

            fig = plot_real_audio(file_bytes, view_type)
            if fig: 
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False})
            else:
                st.warning("Не вдалося відмалювати графік.")
        else: 
            st.info("👈 Завантажте файл зліва, щоб побачити його візуалізацію.")

    st.markdown("---")

    if submit_btn and uploaded_file and watermark_text:
        with st.spinner("Обробка..."):
            try:
                file_bytes = uploaded_file.getvalue()
                filename = uploaded_file.name
                if filename.endswith('.mp3'):
                    audio_segment = AudioSegment.from_file(io.BytesIO(file_bytes), format="mp3")
                    wav_io = io.BytesIO()
                    audio_segment.export(wav_io, format="wav")
                    file_bytes = wav_io.getvalue()
                    filename = filename.replace('.mp3', '.wav')

                algo_str = 'lsb' if 'LSB' in algo_choice else 'dwt_svd'
                data = {'watermark_data': watermark_text, 'algorithm': algo_str, 'use_ecc': use_ecc}
                res = requests.post("http://127.0.0.1:8000/api/protect/", files={'audio': (filename, file_bytes)}, data=data)
                
                if res.status_code == 200:
                    result = res.json()
                    st.success("✅ Файл захищено.")
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown("**ID вашого файлу (натисніть іконку справа, щоб скопіювати):**")
                        st.code(result['track_id'], language=None)
                        st.audio(f"http://127.0.0.1:8000/media/{result['protected_file']}", format="audio/wav")
                    with c2: 
                        st.metric("Швидкодія", f"{result.get('execution_time', 0)} сек")
                        st.metric("Ємність контейнера", result.get('capacity', '0%'))
                    st.session_state.shopping_cart.append({"id": result['track_id'], "name": filename, "watermark": watermark_text, "algorithm": result['algorithm']})
                else:
                    st.error(res.text)
            except Exception as e:
                st.error(e)


elif page == "Експертиза":
    st.title("Експертиза")
    
    col_ext1, col_ext2 = st.columns(2, gap="large")
    with col_ext1: track_id = st.text_input("ID оригінального треку:")
    with col_ext2: suspect_file = st.file_uploader("Завантажте файл (необов'язково, якщо є ID):", type=["wav", "mp3"])

    if not track_id: manual_algo = st.selectbox("Алгоритм для стороннього файлу:", ["DWT+SVD", "LSB"])

    attacks_to_send = []
    with st.expander("Налаштування Стрес-тесту", expanded=True):
        st.markdown("Поставте галочки навпроти тих атак, які хочеш накласти на файл:")
        if st.checkbox("Шум (AWGN)"): 
            attacks_to_send.append({'type': 'noise', 'param': st.slider("SNR (dB):", 0, 40, 20)})
        if st.checkbox("Фільтр"): 
            attacks_to_send.append({'type': 'lowpass', 'param': st.slider("Зріз (Hz):", 500, 8000, 3000)})
        if st.checkbox("Обрізання"): 
            attacks_to_send.append({'type': 'crop', 'param': st.slider("Відсоток з кінця:", 1, 50, 15)})
        if st.checkbox("Зміна гучності"): 
            attacks_to_send.append({'type': 'volume', 'param': st.slider("Коефіцієнт (0.1-тихіше, 2.0-голосніше):", 0.1, 2.0, 0.5)})
        if st.checkbox("Стиснення MP3 (Апаратне)"): 
            attacks_to_send.append({'type': 'mp3', 'param': st.selectbox("Бітрейт:", ["320k", "128k", "64k"])})

    if st.button("Провести експертизу", type="primary", use_container_width=True):
        if not track_id and suspect_file is None:
            st.warning("Введіть ID або завантажте файл.")
        else:
            with st.spinner("Дешифрування..."):
                try:
                    files = {}
                    if suspect_file: 
                        file_bytes = suspect_file.getvalue()
                        filename = suspect_file.name
                        if filename.endswith('.mp3'):
                            audio_segment = AudioSegment.from_file(io.BytesIO(file_bytes), format="mp3")
                            wav_io = io.BytesIO()
                            audio_segment.export(wav_io, format="wav")
                            file_bytes = wav_io.getvalue()
                            filename = filename.replace('.mp3', '.wav')
                        files = {'suspect_audio': (filename, file_bytes)}
                    
                    req_data = {'track_id': track_id.strip(), 'attacks': json.dumps(attacks_to_send), 'manual_algorithm': 'lsb' if not track_id and "LSB" in manual_algo else 'dwt_svd'}
                    res = requests.post("http://127.0.0.1:8000/api/check/", data=req_data, files=files)
                    
                    if res.status_code == 200:
                        data = res.json()
                        st.balloons()
                        
                        st.markdown("### Результати дешифрування")
                        st.caption(f"Алгоритм: **{data.get('algorithm', 'Невідомо')}** | Атаки: **{data.get('attack_applied', 'Без атак')}**")
                        
                        if data.get('ecc_used'):
                            ecc_name = "Хеммінга" if 'lsb' in data.get('algorithm', '') else "Ріда-Соломона"
                            if data.get('attack_applied') != 'Без атак':
                                st.success(f"Спрацював код {ecc_name}. Пошкоджені біти були відновлені.")
                            else:
                                st.info(f"Файл надійно захищено завадостійким кодом {ecc_name}.")
                            
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Оригінал", data.get('original_watermark', 'N/A'))
                        m2.metric("Витягнуто", data.get('extracted_watermark', 'N/A'))
                        
                        match_percent = data.get('match_percentage', 0)
                        if isinstance(match_percent, (int, float)):
                            m3.metric("Збіг (NCC)", f"{match_percent}%")
                        else:
                            m3.metric("Збіг (NCC)", "N/A")
                        
                        st.write("---")
                        st.markdown("##### Фізичні показники та швидкодія")
                        m4, m5, m6, m7 = st.columns(4)
                        m4.metric("SNR", f"{data.get('snr', 'N/A')} dB")
                        m5.metric("PSNR", f"{data.get('psnr', 'N/A')} dB")
                        
                        ber_val = data.get('ber', 'N/A')
                        m6.metric("BER", f"{ber_val}%")
                        m7.metric("Час", f"{data.get('execution_time', 0)} сек")
                        
                        st.write("")
                        report_str = generate_report_content(data, track_id)
                        st.download_button(
                            label="Завантажити звіт (TXT)",
                            data=report_str,
                            file_name=f"Expertise_Report_{datetime.datetime.now().strftime('%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    else: st.error(res.text)
                except Exception as e: st.error(e)


elif page == "Авто-Бенчмарк":
    st.title("Автоматизоване дослідження (Бенчмаркінг)")
    track_id = st.text_input("Введіть ID захищеного треку для аналізу:")
    
    if st.button("Запустити серію експериментів (Це може зайняти до хвилини)", type="primary"):
        if not track_id: 
            st.warning("Будь ласка, введіть дійсний ID треку.")
        else:
            with st.spinner("Виконується бенчмарк..."):
                try:
                    res = requests.post("http://127.0.0.1:8000/api/benchmark/", data={'track_id': track_id.strip()})
                    if res.status_code == 200:
                        results = res.json().get('benchmark_results', [])
                        df = pd.DataFrame(results)
                        
                        st.success("Аналіз успішно завершено!")
                        st.dataframe(df, use_container_width=True)
                        
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Завантажити CSV-таблицю", 
                            data=csv, 
                            file_name="benchmark_results.csv", 
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.error(f"Помилка сервера: {res.json().get('error', res.text)}")
                except Exception as e:
                    st.error(f"Помилка підключення до сервера: {e}")



elif page == "Кошик":
    st.title("Ваш кошик оброблених треків")
    if len(st.session_state.shopping_cart) == 0: 
        st.info("Кошик поки що порожній.")
    else:
        for item in st.session_state.shopping_cart:
            algo_badge = "🟢 DWT+SVD" if 'dwt_svd' in item.get('algorithm', '') else "🟡 LSB"
            ecc_badge = "🛡️ ECC" if '_ecc' in item.get('algorithm', '') else ""
            with st.expander(f"{item['name']} | Знак: {item['watermark']} | {algo_badge} {ecc_badge}"): 
                st.code(item['id'], language="text")
        st.write("")
        if st.button("Очистити кошик", use_container_width=True): 
            st.session_state.shopping_cart = []
            st.rerun()