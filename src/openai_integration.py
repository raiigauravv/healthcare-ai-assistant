"""
Healthcare AI Assistant - OpenAI Integration
Uses gpt-4o for Vision support + Whisper for audio transcription
"""

import os
import io
import base64
import logging
from typing import Dict, Any, Union
from PIL import Image
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISCLAIMER = (
    "You are an AI healthcare assistant for educational purposes only. "
    "You do not provide medical diagnoses. "
    "Always recommend professional medical consultation."
)

_VISION_MODEL = "gpt-4o"        # supports image input
_TEXT_MODEL   = "gpt-4o-mini"   # cheaper for text-only calls


def _to_b64(image: Union[str, Image.Image]) -> str | None:
    """Convert a file path or PIL Image to a base64 JPEG string."""
    try:
        if isinstance(image, str):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Image encoding failed: {e}")
        return None


class OpenAIHealthcareAssistant:
    """OpenAI-powered healthcare assistant with Vision and Whisper support."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found — using mock responses.")
            self.mock_mode = True
            self.client = None
        else:
            self.mock_mode = False
            self.client = OpenAI(api_key=self.api_key)
            logger.info("OpenAI client initialised (gpt-4o Vision + Whisper)")

    # ── Internal helpers ────────────────────────────────────────────────────

    def _text_call(self, prompt: str, max_tokens: int = 800) -> str:
        """Fast text-only call (gpt-4o-mini)."""
        resp = self.client.chat.completions.create(
            model=_TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return resp.choices[0].message.content

    def _vision_call(self, text: str, img_b64: str, max_tokens: int = 600) -> str:
        """Vision call — sends image + text to gpt-4o."""
        resp = self.client.chat.completions.create(
            model=_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                        "detail": "high",
                    }},
                ],
            }],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return resp.choices[0].message.content

    # ── Public methods ──────────────────────────────────────────────────────

    def analyze_medical_image(self, image: Union[str, Image.Image], symptoms: str) -> str:
        """Use GPT-4o Vision to inspect the uploaded image."""
        if self.mock_mode:
            return "Add OPENAI_API_KEY to enable GPT-4o Vision image analysis."
        img_b64 = _to_b64(image)
        if not img_b64:
            return "Unable to encode image for analysis."
        try:
            prompt = (
                f"{DISCLAIMER}\n\n"
                f"Patient symptoms: {symptoms}\n\n"
                "You are examining a medical image. Please provide:\n"
                "1. Detailed visual observations from the image\n"
                "2. Correlation between what you see and the described symptoms\n"
                "3. Recommendations for further professional evaluation\n\n"
                "Be specific about visual findings. Emphasise this is NOT a diagnosis."
            )
            return self._vision_call(prompt, img_b64, 600)
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return "Image analysis unavailable. Please try again."

    def analyze_audio_symptoms(self, audio_path: str, symptoms: str = "") -> str:
        """Transcribe audio using OpenAI Whisper-1."""
        if self.mock_mode:
            return "Add OPENAI_API_KEY to enable Whisper audio transcription."
        if not audio_path or not os.path.exists(audio_path):
            return "Audio file not found."
        try:
            with open(audio_path, "rb") as f:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text",
                )
            text = (transcript or "").strip()
            if text:
                logger.info(f"Whisper transcription: {text[:80]}…")
                return f"Voice transcription (Whisper ASR): {text}"
            return "No speech detected in audio recording."
        except Exception as e:
            logger.error(f"Whisper error: {e}")
            return "Audio transcription unavailable."

    def get_health_recommendation(self, context: str) -> str:
        if self.mock_mode:
            return "Demo mode — add OPENAI_API_KEY secret in HF Space settings."
        try:
            return self._text_call(f"{DISCLAIMER}\n\nPatient question: {context}", 400)
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return "Unable to process your question. Please try again later."

    def comprehensive_health_analysis(
        self,
        symptoms: str,
        image_analysis: str = "",
        audio_analysis: str = "",
        patient_name: str = "",
        patient_age: int = 30,
        patient_gender: str = "Not specified",
    ) -> Dict[str, Any]:
        if self.mock_mode:
            return self._mock_analysis(symptoms, patient_age, patient_gender)
        try:
            prompt = (
                f"{DISCLAIMER}\n\n"
                f"Patient: {patient_name or 'Not provided'}, "
                f"{patient_age} years old, {patient_gender}\n"
                f"Symptoms: {symptoms}\n"
            )
            if image_analysis:
                prompt += f"\nImage Analysis: {image_analysis}"
            if audio_analysis:
                prompt += f"\nVoice/Audio: {audio_analysis}"
            prompt += (
                "\n\nProvide structured analysis:\n"
                "1. Greeting using patient name\n"
                "2. Symptom summary\n"
                "3. Possible conditions to discuss with a doctor\n"
                "4. Health recommendations\n"
                "5. When to seek immediate care"
            )
            text = self._text_call(prompt, 800)
            return self._parse(text)
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                "analysis": "Analysis unavailable. Please try again.",
                "recommendations": "Please consult a healthcare professional.",
                "confidence": 0.0,
            }

    def _parse(self, text: str) -> Dict[str, Any]:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        mid = max(1, len(lines) // 2)
        return {
            "analysis": "\n".join(lines[:mid]),
            "recommendations": "\n".join(lines[mid:]),
            "confidence": min(0.9, max(0.3, len(text) / 1000)),
        }

    def _mock_analysis(self, symptoms: str, age: int, gender: str) -> Dict[str, Any]:
        return {
            "analysis": (
                f"Demo Mode — {age}-year-old {gender}\n"
                f"Symptoms: {symptoms[:100]}\n\n"
                "Add OPENAI_API_KEY in HF Space Secrets for real AI analysis."
            ),
            "recommendations": (
                "• Schedule consultation with a healthcare provider\n"
                "• Monitor symptoms\n"
                "• Seek immediate care if symptoms worsen"
            ),
            "confidence": 0.0,
        }
