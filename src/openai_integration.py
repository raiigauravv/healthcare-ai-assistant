"""
OpenAI Integration for Healthcare AI Assistant
Handles all OpenAI API calls for multimodal healthcare analysis
"""

import os
import httpx
from openai import OpenAI
import base64
import logging
from typing import Dict, Any, Optional
from PIL import Image
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIHealthcareAssistant:
    """
    OpenAI-powered healthcare assistant for multimodal analysis
    """
    
    def __init__(self):
        """Initialize the OpenAI client"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found. Using mock responses.")
            self.mock_mode = True
            self.client = None
        else:
            proxy_url = (
                os.getenv("HTTPS_PROXY")
                or os.getenv("https_proxy")
                or os.getenv("HTTP_PROXY")
                or os.getenv("http_proxy")
            )
            http_client = httpx.Client(proxy=proxy_url) if proxy_url else httpx.Client()
            self.client = OpenAI(api_key=self.api_key, http_client=http_client)
            self.mock_mode = False
            logger.info("OpenAI client initialized successfully")
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for OpenAI Vision API"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Image encoding error: {e}")
            return ""
    
    def analyze_medical_image(self, image_path: str, symptoms: str) -> str:
        """
        Analyze medical image using OpenAI Vision API
        
        Args:
            image_path: Path to the medical image
            symptoms: Patient's described symptoms
            
        Returns:
            Image analysis results
        """
        if self.mock_mode:
            return self._mock_image_analysis(symptoms)
        
        try:
            # Encode image
            base64_image = self.encode_image(image_path)
            if not base64_image:
                return "Unable to process image"
            
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a medical AI assistant analyzing medical images. 
                        Provide professional, accurate observations while emphasizing that this is not a diagnosis 
                        and professional medical consultation is required."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Please analyze this medical image in the context of these symptoms: {symptoms}
                                
                                Provide:
                                1. Objective visual observations
                                2. Possible correlations with described symptoms
                                3. Recommendations for further evaluation
                                
                                Remember to emphasize this is not a medical diagnosis."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return f"Image analysis unavailable: {str(e)}"
    
    def analyze_audio_symptoms(self, audio_path: str, symptoms: str) -> str:
        """
        Analyze audio recording using OpenAI Whisper API
        
        Args:
            audio_path: Path to the audio file
            symptoms: Patient's described symptoms
            
        Returns:
            Audio analysis results
        """
        if self.mock_mode:
            return self._mock_audio_analysis(symptoms)
        
        try:
            # Transcribe audio using Whisper
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            # Analyze the transcript
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a medical AI assistant analyzing patient speech patterns and audio recordings.
                        Look for indicators like breathing patterns, cough characteristics, speech clarity, etc.
                        Provide professional observations while emphasizing this is not a medical diagnosis."""
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze this patient audio transcript in context of their symptoms:
                        
                        Symptoms: {symptoms}
                        Transcript: {transcript.text}
                        
                        Please provide:
                        1. Speech pattern observations
                        2. Audio quality indicators (breathing, cough, clarity)
                        3. Correlation with described symptoms
                        4. Recommendations for professional evaluation
                        
                        Remember: This is not a medical diagnosis."""
                    }
                ],
                max_tokens=400,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Audio analysis error: {e}")
            return f"Audio analysis unavailable: {str(e)}"
    
    def comprehensive_health_analysis(self, 
                                    symptoms: str, 
                                    image_analysis: str = "", 
                                    audio_analysis: str = "",
                                    patient_name: str = "",
                                    patient_age: int = 30,
                                    patient_gender: str = "Not specified") -> Dict[str, Any]:
        """
        Generate comprehensive health analysis using OpenAI GPT-4
        
        Args:
            symptoms: Patient's symptom description
            image_analysis: Results from image analysis
            audio_analysis: Results from audio analysis
            patient_name: Patient's name for personalization
            patient_age: Patient's age
            patient_gender: Patient's gender
            
        Returns:
            Dictionary containing analysis, recommendations, and confidence
        """
        if self.mock_mode:
            return self._mock_comprehensive_analysis(symptoms, patient_age, patient_gender)
        
        try:
            # Ensure proper type handling
            patient_name = str(patient_name) if patient_name is not None else ""
            patient_age = int(patient_age) if patient_age is not None else 30
            patient_gender = str(patient_gender) if patient_gender is not None else "Not specified"
            
            # Construct the analysis prompt with personalization
            greeting = f"Hello {patient_name.strip()}! " if patient_name.strip() else "Hello! "
            
            prompt = f"""
            Patient Information:
            - Name: {patient_name or 'Not provided'}
            - Age: {patient_age}
            - Gender: {patient_gender}
            - Symptoms: {symptoms}
            """
            
            if image_analysis:
                prompt += f"\n- Image Analysis: {image_analysis}"
            
            if audio_analysis:
                prompt += f"\n- Audio Analysis: {audio_analysis}"
            
            # Create personalized greeting if name provided
            greeting = f"Hello {patient_name}, " if patient_name else ""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an AI healthcare assistant providing preliminary analysis based on patient-provided information.{f' Address the patient as {patient_name} when appropriate.' if patient_name else ''} 
                        
                        IMPORTANT DISCLAIMERS:
                        - You do not provide medical diagnoses
                        - Always recommend professional medical consultation
                        - Focus on educational information and general health guidance
                        - Be empathetic but maintain professional boundaries
                        
                        Consider the patient's demographics (age: {patient_age}, gender: {patient_gender}) when relevant to symptoms and conditions.
                        
                        Provide structured analysis with:
                        1. Personalized greeting using the patient's name if provided
                        2. Summary of presented symptoms/data
                        3. Possible conditions to discuss with healthcare providers (considering age/gender when relevant)
                        4. General health recommendations tailored to demographics
                        5. When to seek immediate medical attention
                        6. Confidence level (0-1) based on information completeness"""
                    },
                    {
                        "role": "user",
                        "content": f"""Please analyze this patient case: {prompt}
                        
                        Please provide a comprehensive but responsible analysis following the guidelines above.
                        Start with a personalized greeting if the patient's name is provided.
                        Consider the patient's age and gender in your analysis when medically relevant.
                        Format your response as a structured analysis with clear sections."""
                    }
                ],
                max_tokens=800,
                temperature=0.4
            )
            
            analysis_text = response.choices[0].message.content
            
            # Parse the response and extract sections
            return self._parse_analysis_response(analysis_text)
            
        except Exception as e:
            logger.error(f"Comprehensive analysis error: {e}")
            return {
                "analysis": f"Analysis unavailable due to error: {str(e)}",
                "recommendations": "Please consult with a healthcare professional for proper evaluation.",
                "confidence": 0.0
            }
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the OpenAI response into structured sections"""
        try:
            # Try to extract analysis and recommendations
            lines = response_text.split('\n')
            analysis_lines = []
            recommendation_lines = []
            
            current_section = "analysis"
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for section headers
                if any(keyword in line.lower() for keyword in ['recommendation', 'advice', 'should', 'consider']):
                    current_section = "recommendations"
                
                if current_section == "analysis":
                    analysis_lines.append(line)
                else:
                    recommendation_lines.append(line)
            
            # Estimate confidence based on response completeness
            confidence = min(0.9, max(0.3, len(response_text) / 1000))
            
            return {
                "analysis": '\n'.join(analysis_lines[:10]),  # Limit length
                "recommendations": '\n'.join(recommendation_lines[:8]),
                "confidence": confidence
            }
            
        except Exception:
            return {
                "analysis": response_text[:500],  # Fallback to first 500 chars
                "recommendations": "Please consult with healthcare professionals for proper guidance.",
                "confidence": 0.6
            }
    
    # Mock responses for when OpenAI API is not available
    def _mock_image_analysis(self, symptoms: str) -> str:
        return f"""🔬 Mock Image Analysis:
        
        Visual observations related to symptoms: "{symptoms[:50]}..."
        
        Note: This is a demonstration using mock data. In the actual implementation, 
        this would use OpenAI's GPT-4 Vision API to analyze medical images.
        
        ⚠️ Professional medical imaging interpretation required."""
    
    def _mock_audio_analysis(self, symptoms: str) -> str:
        return f"""🎵 Mock Audio Analysis:
        
        Audio pattern observations for symptoms: "{symptoms[:50]}..."
        
        Note: This is a demonstration using mock data. The actual implementation 
        would use OpenAI's Whisper API for audio transcription and analysis.
        
        ⚠️ Professional medical evaluation recommended."""
    
    def _mock_comprehensive_analysis(self, symptoms: str, age: int, gender: str) -> Dict[str, Any]:
        return {
            "analysis": f"""📋 Comprehensive Health Analysis (Demo Mode):

Patient Profile: {age}-year-old {gender}
Reported Symptoms: {symptoms[:100]}...

🔍 Preliminary Observations:
Based on the provided symptoms, this appears to be a case requiring professional medical evaluation. The combination of symptoms suggests several possible considerations that should be discussed with a healthcare provider.

📊 Information Assessment:
The symptom description provides a good foundation for medical consultation. Additional diagnostic tests may be recommended by healthcare professionals.

⚠️ Important Note: This is a demonstration using mock analysis. Real implementation would use OpenAI GPT-4 for comprehensive medical reasoning.""",
            
            "recommendations": f"""💡 General Recommendations (Demo Mode):

🏥 Immediate Actions:
• Schedule consultation with appropriate healthcare provider
• Monitor symptom progression and note any changes
• Keep a symptom diary for medical appointment

📋 Preparation for Medical Visit:
• List all current medications and supplements
• Prepare questions about symptoms and concerns
• Bring any relevant medical history

🚨 Seek Immediate Care If:
• Symptoms worsen significantly
• New concerning symptoms develop
• You experience severe pain or distress

⚠️ Disclaimer: These are general health recommendations for demonstration purposes. Always follow advice from qualified healthcare professionals.""",
            
            "confidence": 0.75
        }
