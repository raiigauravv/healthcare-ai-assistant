# src/preprocess.py
from PIL import Image
import numpy as np

def preprocess_text(text):
    """Preprocess text input"""
    return text.lower().strip()

def preprocess_image(image):
    """Preprocess PIL image to 224x224 numpy array"""
    image = image.resize((224, 224))
    return np.array(image).astype(np.float32) / 255.0

def preprocess_audio(audio_tuple):
    """Preprocess audio waveform — downmix to mono"""
    waveform, rate = audio_tuple
    if waveform.ndim > 1:
        return waveform.mean(axis=0)
    return waveform
