"""
Healthcare AI Assistant - OpenAI Integration
"""

import os
import logging
from typing import Dict, Any
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISCLAIMER = (
    "You are an AI healthcare assistant for educational purposes only. "
    "You do not provide medical diagnoses. "
    "Always recommend professional medical consultation."
)

_MODEL = "gpt-4o-mini"


class OpenAIHealthcareAssistant:
    """OpenAI-powered healthcare assistant."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found. Using mock responses.")
            self.mock_mode = True
            self.client = None
        else:
            self.mock_mode = False
            self.client = OpenAI(api_key=self.api_key)
            logger.info("OpenAI client initialized successfully")

    def _call(self, prompt: str, max_tokens: int = 800) -> str:
        response = self.client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return response.choices[0].message.content

    def analyze_medical_image(self, image_path: str, symptoms: str) -> str:
        if self.mock_mode:
            return "Mock image analysis. Add OPENAI_API_KEY to enable real analysis."
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
            return "Image analysis unavailable. Please try again."

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
                "analysis": "Analysis unavailable. Please try again.",
                "recommendations": "Please consult a healthcare professional.",
                "confidence": 0.0,
            }

    def get_health_recommendation(self, context: str) -> str:
        if self.mock_mode:
            return "Demo mode — add OPENAI_API_KEY secret in HF Space settings."
        try:
            return self._call(f"{DISCLAIMER}\n\nPatient question: {context}", 300)
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return "Unable to process your question. Please try again later."

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
                "Add OPENAI_API_KEY secret in HF Space settings for real AI analysis."
            ),
            "recommendations": (
                "• Schedule consultation with a healthcare provider\n"
                "• Monitor symptoms\n"
                "• Seek immediate care if symptoms worsen"
            ),
            "confidence": 0.0,
        }
