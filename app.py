import streamlit as st
import torch
import librosa
import soundfile as sf
import noisereduce as nr
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# --------------------
# Load model
# --------------------
model_path = "seemswas/whisper-brunei-asr"

processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --------------------
# Preprocess
# --------------------
def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    y = nr.reduce_noise(y=y, sr=sr)
    output_path = "processed.wav"
    sf.write(output_path, y, sr)
    return output_path

# --------------------
# Transcribe
# --------------------
def transcribe(path):
    audio, sr = librosa.load(path, sr=16000)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(inputs)

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# --------------------
# UI
# --------------------
st.title("Whisper Brunei ASR Demo")

uploaded_file = st.file_uploader("Upload audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    with open("input_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("input_audio.wav")

    if st.button("Run Transcription"):
        st.write("Processing audio...")

        processed_path = preprocess_audio("input_audio.wav")

        st.audio(processed_path)

        original_text = transcribe("input_audio.wav")
        processed_text = transcribe(processed_path)

        st.subheader("Original Audio")
        st.write(original_text)

        st.subheader("Processed Audio")
        st.write(processed_text)