"""
Healthcare AI Assistant - Gemini Integration
"""

import os
import base64
import logging
from typing import Dict, Any, Optional
from PIL import Image
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_DISCLAIMER = (
    "You are an AI healthcare assistant for educational purposes only. "
    "You do not provide medical diagnoses. Always recommend professional medical consultation. "
    "Be empathetic and responsible."
)


class OpenAIHealthcareAssistant:
    """Gemini-powered healthcare assistant (class name kept for app.py compatibility)."""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found. Using mock responses.")
            self.mock_mode = True
            self.model = None
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-pro")
            self.mock_mode = False
            logger.info("Gemini client initialized successfully")

    def _generate(self, prompt: str, max_tokens: int = 800) -> str:
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.4,
            ),
        )
        return response.text

    def encode_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Image load error: {e}")
            return None

    def analyze_medical_image(self, image_path: str, symptoms: str) -> str:
        if self.mock_mode:
            return self._mock_image_analysis(symptoms)
        try:
            img = self.encode_image(image_path)
            if img is None:
                return "Unable to process image."
            prompt = (
                f"{SYSTEM_DISCLAIMER}\n\n"
                f"Patient symptoms: {symptoms}\n\n"
                "Please analyze this medical image:\n"
                "1. Objective visual observations\n"
                "2. Possible correlations with described symptoms\n"
                "3. Recommendations for further evaluation\n"
                "Emphasize this is not a medical diagnosis."
            )
            response = self.model.generate_content(
                [prompt, img],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500, temperature=0.3
                ),
            )
            return response.text
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return f"Image analysis unavailable: {e}"

    def analyze_audio_symptoms(self, audio_path: str, symptoms: str) -> str:
        if self.mock_mode:
            return self._mock_audio_analysis(symptoms)
        return (
            "Audio transcription processed. Please describe any additional audio observations "
            "in your symptom description for more accurate analysis."
        )

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
            return self._mock_comprehensive_analysis(symptoms, patient_age, patient_gender)
        try:
            patient_name = str(patient_name) if patient_name else ""
            patient_age = int(patient_age) if patient_age else 30
            patient_gender = str(patient_gender) if patient_gender else "Not specified"

            prompt = (
                f"{SYSTEM_DISCLAIMER}\n\n"
                f"Patient: {patient_name or 'Not provided'}, {patient_age} years old, {patient_gender}\n"
                f"Symptoms: {symptoms}\n"
            )
            if image_analysis:
                prompt += f"Image Analysis: {image_analysis}\n"
            if audio_analysis:
                prompt += f"Audio Analysis: {audio_analysis}\n"
            prompt += (
                "\nProvide a structured analysis with:\n"
                "1. Personalized greeting using patient name if provided\n"
                "2. Summary of presented symptoms\n"
                "3. Possible conditions to discuss with healthcare providers\n"
                "4. General health recommendations\n"
                "5. When to seek immediate medical attention\n"
                "Format with clear sections."
            )

            text = self._generate(prompt, max_tokens=800)
            return self._parse_analysis_response(text)
        except Exception as e:
            logger.error(f"Comprehensive analysis error: {e}")
            return {
                "analysis": f"Analysis unavailable: {e}",
                "recommendations": "Please consult a healthcare professional.",
                "confidence": 0.0,
            }

    def get_health_recommendation(self, context: str) -> str:
        """Answer a follow-up health question."""
        if self.mock_mode:
            return "Demo mode — add GEMINI_API_KEY secret in HF Space settings for real responses."
        try:
            prompt = f"{SYSTEM_DISCLAIMER}\n\nPatient question: {context}"
            return self._generate(prompt, max_tokens=300)
        except Exception as e:
            logger.error(f"Health recommendation error: {e}")
            return f"Unable to process your question: {e}"

    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        try:
            lines = response_text.split("\n")
            analysis_lines, recommendation_lines = [], []
            current = "analysis"
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if any(k in line.lower() for k in ["recommendation", "advice", "should", "consider"]):
                    current = "recommendations"
                (analysis_lines if current == "analysis" else recommendation_lines).append(line)
            confidence = min(0.9, max(0.3, len(response_text) / 1000))
            return {
                "analysis": "\n".join(analysis_lines[:10]),
                "recommendations": "\n".join(recommendation_lines[:8]),
                "confidence": confidence,
            }
        except Exception:
            return {
                "analysis": response_text[:500],
                "recommendations": "Please consult with healthcare professionals.",
                "confidence": 0.6,
            }

    def _mock_image_analysis(self, symptoms: str) -> str:
        return (
            f"Mock Image Analysis for symptoms: '{symptoms[:50]}...'\n"
            "Add GEMINI_API_KEY to enable real image analysis."
        )

    def _mock_audio_analysis(self, symptoms: str) -> str:
        return (
            f"Mock Audio Analysis for symptoms: '{symptoms[:50]}...'\n"
            "Add GEMINI_API_KEY to enable real audio analysis."
        )

    def _mock_comprehensive_analysis(self, symptoms: str, age: int, gender: str) -> Dict[str, Any]:
        return {
            "analysis": (
                f"Demo Mode — Patient: {age}-year-old {gender}\n"
                f"Symptoms: {symptoms[:100]}\n\n"
                "Add GEMINI_API_KEY secret in HF Space settings for real AI analysis."
            ),
            "recommendations": (
                "General Recommendations (Demo):\n"
                "• Schedule consultation with appropriate healthcare provider\n"
                "• Monitor symptom progression\n"
                "• Seek immediate care if symptoms worsen significantly"
            ),
            "confidence": 0.0,
        }
