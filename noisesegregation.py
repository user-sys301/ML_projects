import os
import torch
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr

from demucs.pretrained import get_model
from demucs.apply import apply_model

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

INPUT_AUDIO = "mixed_audio.wav"
OUTPUT_AUDIO = "clean_separated_voice.wav"

TARGET_SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------
# LOAD AUDIO
# ---------------------------------------------------

print("[INFO] Loading audio...")

audio, sr = librosa.load(INPUT_AUDIO, sr=TARGET_SR, mono=True)

# ---------------------------------------------------
# ADVANCED NOISE REDUCTION
# ---------------------------------------------------

print("[INFO] Applying adaptive noise reduction...")

reduced_noise = nr.reduce_noise(
    y=audio,
    sr=sr,
    stationary=False,
    prop_decrease=0.9
)

# ---------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------

reduced_noise = librosa.util.normalize(reduced_noise)

# ---------------------------------------------------
# CONVERT TO TORCH TENSOR
# ---------------------------------------------------

waveform = torch.tensor(reduced_noise).float()

# Demucs expects [batch, channels, samples]
waveform = waveform.unsqueeze(0).unsqueeze(0).to(DEVICE)

# ---------------------------------------------------
# LOAD DEMUCS MODEL
# ---------------------------------------------------

print("[INFO] Loading Demucs model...")

model = get_model(name="htdemucs")
model.to(DEVICE)
model.eval()

# ---------------------------------------------------
# SOURCE SEPARATION
# ---------------------------------------------------

print("[INFO] Performing AI source separation...")

with torch.no_grad():
    sources = apply_model(model, waveform)

# ---------------------------------------------------
# VOCAL EXTRACTION
# ---------------------------------------------------

# Demucs output:
# [drums, bass, other, vocals]

vocals = sources[0, 3].cpu().numpy()

# ---------------------------------------------------
# POST PROCESSING
# ---------------------------------------------------

print("[INFO] Post-processing vocals...")

# Remove residual noise
vocals_clean = nr.reduce_noise(
    y=vocals,
    sr=sr,
    stationary=False,
    prop_decrease=0.8
)

# Final normalization
vocals_clean = librosa.util.normalize(vocals_clean)

# ---------------------------------------------------
# SAVE OUTPUT
# ---------------------------------------------------

sf.write(OUTPUT_AUDIO, vocals_clean, sr)

print(f"[SUCCESS] Clean voice saved to: {OUTPUT_AUDIO}")