"""
Healthcare AI Assistant - Gemini REST API (no SDK, avoids websockets conflict)
"""

import os
import logging
import requests
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/models/gemini-1.5-flash:generateContent"
)

DISCLAIMER = (
    "You are an AI healthcare assistant for educational purposes only. "
    "You do not provide medical diagnoses. "
    "Always recommend professional medical consultation."
)


def _sanitize_error(e: Exception) -> str:
    """Return a safe error string with no API key or URL."""
    msg = str(e)
    # Strip anything that looks like a URL containing a key= param
    import re
    msg = re.sub(r"https?://[^\s]*key=[^\s]*", "<Gemini API>", msg)
    return msg


def _call_gemini(prompt: str, api_key: str, max_tokens: int = 800) -> str:
    resp = requests.post(
        f"{_GEMINI_URL}?key={api_key}",
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.4,
            },
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


class OpenAIHealthcareAssistant:
    """Gemini-powered healthcare assistant (class name kept for app.py compatibility)."""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found. Using mock responses.")
            self.mock_mode = True
        else:
            self.mock_mode = False
            logger.info("Gemini API key loaded successfully")

    def _call(self, prompt: str, max_tokens: int = 800) -> str:
        return _call_gemini(prompt, self.api_key, max_tokens)

    def analyze_medical_image(self, image_path: str, symptoms: str) -> str:
        if self.mock_mode:
            return f"Mock image analysis. Add GEMINI_API_KEY to enable real analysis."
        try:
            prompt = (
                f"{DISCLAIMER}\n\nPatient symptoms: {symptoms}\n\n"
                "Provide observations related to these symptoms:\n"
                "1. Possible findings\n"
                "2. Correlations with described symptoms\n"
                "3. Recommendations for further evaluation\n"
                "Emphasize this is not a medical diagnosis."
            )
            return self._call(prompt, 500)
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return f"Image analysis unavailable: {_sanitize_error(e)}"

    def analyze_audio_symptoms(self, audio_path: str, symptoms: str) -> str:
        return "Audio recorded. Include any additional details in your symptom description."

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
            patient_name = str(patient_name) if patient_name else ""
            patient_age = int(patient_age) if patient_age else 30
            patient_gender = str(patient_gender) if patient_gender else "Not specified"
            prompt = (
                f"{DISCLAIMER}\n\n"
                f"Patient: {patient_name or 'Not provided'}, "
                f"{patient_age} years old, {patient_gender}\n"
                f"Symptoms: {symptoms}\n"
            )
            if image_analysis:
                prompt += f"Image Analysis: {image_analysis}\n"
            if audio_analysis:
                prompt += f"Audio Analysis: {audio_analysis}\n"
            prompt += (
                "\nProvide structured analysis:\n"
                "1. Greeting using patient name\n"
                "2. Symptom summary\n"
                "3. Possible conditions to discuss with a doctor\n"
                "4. Health recommendations\n"
                "5. When to seek immediate care"
            )
            text = self._call(prompt, 800)
            return self._parse(text)
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                "analysis": f"Analysis unavailable: {_sanitize_error(e)}",
                "recommendations": "Please consult a healthcare professional.",
                "confidence": 0.0,
            }

    def get_health_recommendation(self, context: str) -> str:
        if self.mock_mode:
            return "Demo mode — add GEMINI_API_KEY secret in HF Space settings."
        try:
            return self._call(f"{DISCLAIMER}\n\nPatient question: {context}", 300)
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return f"Unable to process your question: {_sanitize_error(e)}"

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
                "Add GEMINI_API_KEY secret in HF Space settings for real AI analysis."
            ),
            "recommendations": (
                "• Schedule consultation with a healthcare provider\n"
                "• Monitor symptoms\n"
                "• Seek immediate care if symptoms worsen"
            ),
            "confidence": 0.0,
        }
