import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="perth")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

import streamlit as st
import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from pydub import AudioSegment
import io
import os
import uuid
import shutil
import numpy as np
import time
import atexit

_ffmpeg_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
if _ffmpeg_path:
    AudioSegment.converter = _ffmpeg_path

@st.cache_resource
def load_model():
    device = "cpu"
    return ChatterboxTTS.from_pretrained(device=device)

model = load_model()
try: model.eval()
except: pass
try: torch.set_num_threads(4)
except: pass

SAMPLE_SR = 24000
SAMPLE_VOICES = {
    "Male - Calm Narrator": "./voices/male_1_5s.wav",
    "Male - Energetic Host": "./voices/male_2_5s.wav",
    "Female - Warm Teacher": "./voices/female_1_5s.wav",
}

def _save_wav(path, wav, sr):
    if isinstance(wav, list): wav = np.array(wav, dtype=np.float32)
    if isinstance(wav, np.ndarray):
        if wav.ndim == 1: wav = np.expand_dims(wav, 0)
        elif wav.ndim == 2 and wav.shape[0] < wav.shape[1] and wav.shape[0] != 1: wav = wav.T
        wav = torch.from_numpy(wav)
    if not torch.is_tensor(wav): wav = torch.tensor(wav, dtype=torch.float32)
    ta.save(path, wav.to(dtype=torch.float32), sr)

def generate_voiceover(text, voice_path, exaggeration=0.5, cfg=0.5):
    temp_wav = f"temp_{uuid.uuid4().hex}.wav"
    start_ts = time.time()
    try:
        with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
            try: wav = model.generate(text, audio_prompt_path=voice_path, exaggeration=exaggeration, cfg=cfg)
            except TypeError: wav = model.generate(text, audio_prompt_path=voice_path, exaggeration=exaggeration)
    except Exception as e: st.error(f"Model generation failed: {e}"); st.stop()
    _save_wav(temp_wav, wav, SAMPLE_SR)
    if "temp_files" not in st.session_state: st.session_state["temp_files"] = []
    st.session_state["temp_files"].append(temp_wav)
    return AudioSegment.from_wav(temp_wav)

st.set_page_config(page_title="BrandVoice Studio", layout="centered")
st.markdown("""
<style>
    .stButton>button {background-color:#4B6BFB;color:white;height:3em;width:100%;}
    .stSlider>div>div>div>div{color:#4B6BFB;}
    .stTextArea>div>textarea{border-radius:10px;}
</style>
""", unsafe_allow_html=True)
st.title("🎙️ BrandVoice Studio")
st.markdown("Consistent, professional voiceovers for marketing, demos, social media & customer service.")
if not _ffmpeg_path: st.warning("⚠️ ffmpeg not found. MP3 export may fail.")

if "step" not in st.session_state: st.session_state.step = 1

def show_modal(title, message):
    st.warning(f"⚠️ {title}: {message}")

if st.session_state.step == 1:
    st.header("Step 1: Enter Your Script")
    script = st.text_area("Paste text (≥250 words recommended)", height=220, placeholder="Write your marketing script, explainer, or demo text...")
    if st.button("Next: Pick Voice & Emotion"):
        words = len(script.strip().split())
        if words < 10: show_modal("Too Short", "Please enter at least 10 words."); st.stop()
        if words > 500: st.info("💡 Tip: Text >500 words may take longer to synthesize.")
        st.session_state.script = script
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.header("Step 2: Choose Voice & Emotion")
    col1, col2 = st.columns(2)
    with col1:
        selected_voice = st.selectbox("Pick a voice", list(SAMPLE_VOICES.keys()))
        voice_path = SAMPLE_VOICES[selected_voice]
        uploaded = st.file_uploader("Or upload a 5s voice clip to clone", type=['wav','mp3'])
        if uploaded:
            temp_upload = f"temp_clone_{uuid.uuid4().hex}.wav"
            with open(temp_upload, "wb") as f: f.write(uploaded.getvalue())
            voice_path = temp_upload
    with col2:
        emotion = st.slider("Emotion intensity", 0.0,1.0,0.5, help="Low=Calm | High=Excited/Dramatic")
        if st.button("Neutral"): emotion=0.3; st.rerun()
        if st.button("Energetic"): emotion=0.6; st.rerun()
        if st.button("Dramatic"): emotion=0.9; st.rerun()
    if st.button("Next: Generate"):
        if not voice_path: show_modal("Missing Voice", "Please select or upload a voice clip."); st.stop()
        st.session_state.voice_path = voice_path
        st.session_state.emotion = emotion
        st.session_state.step = 3
        st.rerun()

elif st.session_state.step == 3:
    st.header("Step 3: Generate Voiceover")
    if st.button("🚀 Generate Audio"):
        with st.spinner("Synthesizing audio... please wait."):
            audio = generate_voiceover(st.session_state.script, st.session_state.voice_path, st.session_state.emotion)
            mp3_buffer = io.BytesIO()
            audio.export(mp3_buffer, format="mp3")
            mp3_buffer.seek(0)
            st.success("✅ Voiceover ready!")
            st.download_button("📥 Download MP3", mp3_buffer.getvalue(), file_name="voiceover.mp3", mime="audio/mpeg")
            st.audio(mp3_buffer.getvalue(), format="audio/mp3")
    if st.button("Back to Customize"): st.session_state.step=2; st.rerun()

with st.sidebar:
    if st.button("🔄 New Project"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.session_state.step=1; st.rerun()
    st.markdown("---")
    st.info("💡 Tips:\n- Use ≥250 words for best results.\n- Upload a 5s clip to clone brand voice.\n- CPU optimized; long scripts may take a few minutes.")

def cleanup():
    files_to_remove = []
    try: files_to_remove.extend(st.session_state.get("temp_files",[]))
    except: pass
    for f in os.listdir("."):
        if f.startswith("temp_") and f.endswith(".wav"): files_to_remove.append(f)
    for f in set(files_to_remove):
        try: os.remove(f)
        except: pass

atexit.register(cleanup)
