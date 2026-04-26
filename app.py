import streamlit as st
import torch
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
from scipy import signal
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# --------------------
# Load model
# --------------------
model_path = "seemswas/whisper-brunei-asr"

processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.generation_config.task = "transcribe"

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --------------------
# Preprocess 
# --------------------
def preprocess_audio(audio_path, speed_factor=1.0):
    """
    Audio preprocessing pipeline:
    - Speed control
    - Bandpass filter
    - Silence removal
    - Noise reduction
    - Normalization
    """
    y, sr = librosa.load(audio_path, sr=16000)

    # --- Speed control ---
    if speed_factor != 1.0:
        try:
            y = librosa.effects.time_stretch(y, rate=speed_factor)
        except Exception as e:
            st.warning(f"Time-stretch failed: {e}")
            new_sr = int(sr * speed_factor)
            y = librosa.resample(y, orig_sr=sr, target_sr=new_sr)
            sr = new_sr

    # --- Bandpass filter ---
    lowcut = 100.0
    highcut = 7900.0
    nyquist = sr / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = signal.butter(6, [low, high], btype='band')
    y = signal.filtfilt(b, a, y)

    # --- Silence removal ---
    intervals = librosa.effects.split(y, top_db=25)
    if len(intervals) > 0:
        y = np.concatenate([y[i[0]:i[1]] for i in intervals])

    # --- Noise reduction ---
    y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)

    # --- Normalization ---
    rms = np.sqrt(np.mean(y**2)) + 1e-9
    target_dBFS = -20
    scalar = 10 ** (target_dBFS / 20) / rms
    y = y * scalar

    output_path = "processed.wav"
    sf.write(output_path, y, sr)

    return output_path

# --------------------
# Transcribe
# --------------------
def transcribe(path):
    audio, sr = librosa.load(path, sr=16000)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs,
            task="transcribe"   
        )

    return processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

# --------------------
# UI
# --------------------
st.title("🎙 Whisper Brunei ASR Demo")

uploaded_file = st.file_uploader("Upload audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    with open("input_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("input_audio.wav")

    # --- Speed control UI ---
    st.subheader("Playback Speed")

    speed_option = st.radio(
        "Choose speed:",
        ["0.5x (Slow)", "1x (Normal)", "2x (Fast)"]
    )

    if speed_option == "0.5x (Slow)":
        speed_factor = 0.5
    elif speed_option == "2x (Fast)":
        speed_factor = 2.0
    else:
        speed_factor = 1.0

    if st.button("Run Transcription"):
        st.write("Processing audio...")

        processed_path = preprocess_audio("input_audio.wav", speed_factor)

        st.audio(processed_path)

        original_text = transcribe("input_audio.wav")
        processed_text = transcribe(processed_path)

        st.subheader("Original Audio")
        st.write(original_text)

        st.subheader("Processed Audio")
        st.write(processed_text)