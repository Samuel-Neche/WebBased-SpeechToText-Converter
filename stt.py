import streamlit as st
import os
import torch
import torchaudio
import librosa
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydub import AudioSegment
from streamlit_webrtc import WebRtcMode
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import av
import numpy as np
import tempfile
from datasets import load_from_disk
import pandas as pd

import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# === Constants ===
MODEL_DIR = "./whisper_multilingual"
LANGUAGES = {"Yoruba": "yo", "Hausa": "ha", "Igbo": "ig"}
PROCESSED_DIR = "./preprocessed_languages_datasets"

st.title("Multilingual Speech-to-Text App")
st.sidebar.title("Choose Language")
selected_language = st.sidebar.radio("Available Languages", list(LANGUAGES.keys()))
lang_code = LANGUAGES[selected_language]

@st.cache_resource
def load_model(lang_code):
    model_path = os.path.join(MODEL_DIR, lang_code)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at {model_path}")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    return processor, model

try:
    processor, model = load_model(lang_code)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def convert_to_mp3(input_path, output_path="temp.mp3"):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="mp3")
    return output_path

def transcribe_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        pred_ids = model.generate(inputs)
        return processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

# === File Upload Option ===
st.header("Upload Audio")
uploaded = st.file_uploader("Upload audio (wav, mp3, m4a, ogg)", type=["wav", "mp3", "m4a", "ogg"])
if uploaded:
    temp_file_path = os.path.join(tempfile.gettempdir(), uploaded.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded.read())
    mp3_path = convert_to_mp3(temp_file_path)
    transcript = transcribe_audio(mp3_path)
    st.success("Transcription:")
    st.write(transcript)

# === Live Mic Recording Option ===
st.header("Record from Microphone")

class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.audio_data = b""

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, samplerate=frame.sample_rate)
            try:
                transcript = transcribe_audio(f.name)
                st.session_state["live_transcript"] = transcript
            except Exception as e:
                st.session_state["live_transcript"] = f"Transcription error: {e}"
        return frame

ctx = webrtc_streamer(
    key="mic",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
)

if "live_transcript" in st.session_state:
    st.success("Live Transcription:")
    st.write(st.session_state["live_transcript"])

# === Dataset Visualization Section ===
st.header("Dataset Visualization")

split_option = st.selectbox("Choose Split", ["train", "test"])
lang_option = st.selectbox("Choose Language for Dataset Stats", list(LANGUAGES.keys()))
lang_code_ds = LANGUAGES[lang_option]

ds_path = os.path.join(PROCESSED_DIR, lang_code_ds, split_option)

if os.path.exists(ds_path):
    try:
        dataset = load_from_disk(ds_path)
        st.success(f"{lang_option} {split_option} dataset loaded successfully.")

        # Convert to DataFrame
        df = dataset.to_pandas()
        st.subheader("Dataset Sample")
        st.dataframe(df[["sentence"]].head(10))

        st.subheader("Dataset Stats")
        st.write(f"Number of Samples: {len(df)}")

        if "audio" in df.columns:
            durations = df["audio"].apply(lambda a: len(a["array"]) / a["sampling_rate"])
            st.write(f"Average Duration (s): {durations.mean():.2f}")
            st.write(f"Total Duration (min): {durations.sum() / 60:.2f}")
            st.line_chart(durations[:100], height=200)

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
else:
    st.warning(f"Dataset for {lang_option} - {split_option} not found at {ds_path}")
