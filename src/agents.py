# src/agents.py - Healthcare AI Specialist Agents
import os
from typing import Dict, List, Optional, Union
import logging
from PIL import Image
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class BaseHealthcareAgent:
    """Base class for all healthcare specialist agents"""
    
    def __init__(self, specialty: str, system_prompt: str):
        self.specialty = specialty
        self.system_prompt = system_prompt
        self.client = None  # unused; AgentCoordinator handles all API calls
        
    async def analyze(self, patient_data: Dict) -> Dict:
        """Base analysis method - to be overridden by specialist agents"""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._format_patient_data(patient_data)}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=800,
                temperature=0.4
            )
            
            return {
                "agent": self.specialty,
                "analysis": response.choices[0].message.content,
                "confidence": "moderate",  # Could be enhanced with actual confidence scoring
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in {self.specialty} agent analysis: {e}")
            return {
                "agent": self.specialty,
                "analysis": f"Error occurred during {self.specialty} analysis: {str(e)}",
                "confidence": "low",
                "status": "error"
            }
    
    def _format_patient_data(self, patient_data: Dict) -> str:
        """Format patient data for the AI prompt"""
        formatted = f"""
Patient Information:
- Name: {patient_data.get('name', 'Not provided')}
- Age: {patient_data.get('age', 'Not provided')}
- Gender: {patient_data.get('gender', 'Not provided')}
- Symptoms: {patient_data.get('symptoms', 'Not provided')}
"""
        
        if patient_data.get('has_image'):
            formatted += "- Medical image: Provided for analysis\n"
        if patient_data.get('has_audio'):
            formatted += "- Audio description: Provided\n"
            
        return formatted


class TriageAgent(BaseHealthcareAgent):
    """Agent specialized in initial assessment and urgency classification"""
    
    def __init__(self):
        system_prompt = """You are a highly experienced medical triage nurse with 20+ years of experience. 
        Your role is to:
        1. Assess the urgency level of patient symptoms (Emergency, Urgent, Semi-urgent, Non-urgent)
        2. Provide initial recommendations for immediate care
        3. Determine which medical specialists should be consulted
        4. Identify red flags that require immediate medical attention

        URGENCY LEVELS:
        - EMERGENCY: Life-threatening conditions requiring immediate 911/emergency care
        - URGENT: Conditions requiring same-day medical attention
        - SEMI-URGENT: Should see doctor within 1-3 days
        - NON-URGENT: Can wait for routine appointment

        Always provide clear, actionable guidance while emphasizing when professional medical care is needed.
        Include specific next steps and timeframes for seeking care."""
        
        super().__init__("triage", system_prompt)
    
    async def analyze(self, patient_data: Dict) -> Dict:
        """Specialized triage analysis with urgency assessment"""
        result = await super().analyze(patient_data)
        
        # Extract urgency level from the analysis with improved logic
        analysis_text = result.get("analysis", "").lower()
        
        # Priority-based urgency detection (most severe first)
        if any(keyword in analysis_text for keyword in ["emergency", "911", "call 911", "immediate", "life-threatening"]):
            urgency = "EMERGENCY"
        elif any(keyword in analysis_text for keyword in ["semi-urgent", "semi urgent"]):
            urgency = "SEMI-URGENT"
        elif "urgent" in analysis_text and "non-urgent" not in analysis_text and "semi-urgent" not in analysis_text:
            urgency = "URGENT"
        elif "non-urgent" in analysis_text:
            urgency = "NON-URGENT"
        else:
            # Default based on symptom severity keywords
            severe_keywords = ["severe", "intense", "difficulty breathing", "chest pain", "bleeding"]
            if any(keyword in analysis_text for keyword in severe_keywords):
                urgency = "URGENT"
            else:
                urgency = "SEMI-URGENT"  # Conservative default for unclear cases
        
        result["urgency"] = urgency
        result["specialties_recommended"] = self._extract_specialties(analysis_text)
        
        return result
    
    def _extract_specialties(self, analysis_text: str) -> List[str]:
        """Extract recommended medical specialties from analysis"""
        specialties = []
        specialty_keywords = {
            "dermatology": ["skin", "rash", "dermatologist", "dermatology"],
            "cardiology": ["heart", "chest pain", "cardiology", "cardiologist"],
            "neurology": ["headache", "neurological", "neurology", "neurologist"],
            "orthopedics": ["bone", "joint", "orthopedic", "fracture"],
            "gastroenterology": ["stomach", "digestive", "gastroenterology"],
            "psychiatry": ["mental", "anxiety", "depression", "psychiatric"]
        }
        
        for specialty, keywords in specialty_keywords.items():
            if any(keyword in analysis_text for keyword in keywords):
                specialties.append(specialty)
        
        return specialties


class DermatologyAgent(BaseHealthcareAgent):
    """Agent specialized in skin conditions and dermatological analysis"""
    
    def __init__(self):
        system_prompt = """You are a board-certified dermatologist with expertise in:
        1. Skin condition diagnosis and assessment
        2. Analysis of skin lesions, rashes, and abnormalities
        3. Recommendations for skincare and treatment
        4. Recognition of concerning skin changes that need immediate attention

        When analyzing skin conditions:
        - Describe what you observe in detail
        - Provide differential diagnosis possibilities
        - Recommend appropriate next steps (topical treatments, specialist referral, etc.)
        - Identify any concerning features that warrant urgent evaluation
        - Consider patient demographics (age, gender) in your assessment

        Always emphasize that visual examination by a dermatologist is essential for accurate diagnosis.
        Focus on education and reassurance while being thorough about when to seek professional care."""
        
        super().__init__("dermatology", system_prompt)
    
    async def analyze_with_image(self, patient_data: Dict, image: Optional[Image.Image] = None) -> Dict:
        """Enhanced analysis that can include image analysis"""
        if image:
            # Use GPT-4 Vision for image analysis
            try:
                import base64
                import io
                
                # Convert PIL Image to base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{self._format_patient_data(patient_data)}\n\nPlease analyze the provided medical image of the skin condition."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ]
                
                response = self.client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=messages,
                    max_tokens=800,
                    temperature=0.4
                )
                
                return {
                    "agent": self.specialty,
                    "analysis": response.choices[0].message.content,
                    "confidence": "high",  # Higher confidence with image
                    "status": "success",
                    "image_analyzed": True
                }
                
            except Exception as e:
                logger.error(f"Error in image analysis: {e}")
                # Fall back to text-only analysis
                return await super().analyze(patient_data)
        else:
            return await super().analyze(patient_data)


class GeneralPracticeAgent(BaseHealthcareAgent):
    """Agent providing general medical assessment and holistic care recommendations"""
    
    def __init__(self):
        system_prompt = """You are an experienced family medicine physician providing comprehensive primary care.
        Your approach includes:
        1. Holistic assessment considering all patient factors
        2. Differential diagnosis with common conditions prioritized
        3. Lifestyle and preventive care recommendations
        4. Coordination of care with specialists when needed
        5. Patient education and health promotion

        Consider the whole patient - physical, mental, and social factors.
        Provide practical, evidence-based recommendations while emphasizing the importance of 
        ongoing relationship with a primary care provider.
        
        Focus on:
        - Most likely diagnoses given the presentation
        - Self-care measures that are safe and appropriate
        - When to seek medical care and what type
        - Preventive measures and lifestyle modifications
        - Red flags that require immediate attention"""
        
        super().__init__("general_practice", system_prompt)


class FollowUpAgent(BaseHealthcareAgent):
    """Agent specialized in handling follow-up questions and clarifications"""
    
    def __init__(self):
        system_prompt = """You are a patient care coordinator and health educator specialized in:
        1. Clarifying medical recommendations and instructions
        2. Answering follow-up questions about diagnoses and treatments
        3. Providing additional health education
        4. Helping patients understand when to seek further care
        5. Offering emotional support and reassurance

        You maintain the context of previous medical consultations and provide clarification
        in simple, understandable language. You help patients feel confident about their
        care plan while knowing when to escalate concerns to healthcare providers."""
        
        super().__init__("follow_up", system_prompt)
    
    async def answer_followup(self, original_analysis: str, patient_question: str, patient_data: Dict) -> Dict:
        """Handle follow-up questions based on previous analysis"""
        enhanced_prompt = f"""
        Original Medical Analysis:
        {original_analysis}
        
        Patient's Follow-up Question:
        {patient_question}
        
        Patient Information:
        {self._format_patient_data(patient_data)}
        
        Please provide a clear, helpful response that addresses their specific question
        while maintaining consistency with the original analysis.
        """
        
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": enhanced_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=600,
                temperature=0.3
            )
            
            return {
                "agent": self.specialty,
                "response": response.choices[0].message.content,
                "type": "follow_up",
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in follow-up response: {e}")
            return {
                "agent": self.specialty,
                "response": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "type": "follow_up",
                "status": "error"
            }


SPECIALISTS = {
    "General Physician": (
        "an experienced family medicine physician. Provide: "
        "1) Assessment of likely conditions given the symptoms. "
        "2) Self-care recommendations. "
        "3) When to seek medical care. "
        "This is for educational purposes only — always recommend professional consultation."
    ),
    "Cardiologist": (
        "a cardiologist specializing in heart and cardiovascular conditions. Provide: "
        "1) Cardiovascular assessment relevant to the symptoms. "
        "2) Recommendations including lifestyle factors. "
        "3) Red flags requiring urgent cardiac evaluation. "
        "This is for educational purposes only — always recommend professional consultation."
    ),
    "Neurologist": (
        "a neurologist specializing in brain and nervous system conditions. Provide: "
        "1) Neurological assessment relevant to the symptoms. "
        "2) Recommendations and next steps. "
        "3) Warning signs needing urgent neurological evaluation. "
        "This is for educational purposes only — always recommend professional consultation."
    ),
    "Dermatologist": (
        "a dermatologist specializing in skin conditions. Provide: "
        "1) Dermatological assessment relevant to the symptoms. "
        "2) Skincare and treatment recommendations. "
        "3) Any concerning features requiring urgent evaluation. "
        "This is for educational purposes only — always recommend professional consultation."
    ),
}


class AgentCoordinator:
    """Coordinates multiple agents and manages the overall analysis workflow"""

    def __init__(self):
        self.triage_agent = TriageAgent()
        self.dermatology_agent = DermatologyAgent()
        self.general_practice_agent = GeneralPracticeAgent()
        self.followup_agent = FollowUpAgent()

        self._api_key = os.getenv("GEMINI_API_KEY")

    def analyze_with_agents(self, patient_context: Dict, symptoms_text: str,
                            image_data=None, audio_data=None) -> Dict:
        """Synchronous analysis by all 4 specialist agents."""
        if not self._api_key:
            return {
                name: {
                    "analysis": "GEMINI_API_KEY not configured.",
                    "recommendations": "Add GEMINI_API_KEY as a secret in HF Space settings.",
                    "confidence_score": 0.0,
                }
                for name in SPECIALISTS
            }

        patient_line = (
            f"Patient: {patient_context.get('name', 'Patient')}, "
            f"{patient_context.get('age', '?')} year old "
            f"{patient_context.get('gender', '')}.\n"
            f"Symptoms: {symptoms_text}"
        )

        results = {}
        for name, role in SPECIALISTS.items():
            results[name] = self._call_specialist(name, role, patient_line)
        return results

    def _call_specialist(self, name: str, role: str, patient_line: str) -> Dict:
        try:
            prompt = (
                f"You are {role}\n\n"
                f"Analyze this patient case and provide:\n"
                f"1. Analysis of likely conditions\n"
                f"2. Specific recommendations\n"
                f"3. Red flags to watch for\n\n"
                f"{patient_line}\n\n"
                f"This is for educational purposes only. Always recommend professional medical consultation."
            )
            resp = requests.post(
                "https://generativelanguage.googleapis.com/v1beta"
                f"/models/gemini-2.0-flash:generateContent?key={self._api_key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 600, "temperature": 0.4},
                },
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            lines = text.split("\n")
            mid = max(1, len(lines) // 2)
            return {
                "analysis": "\n".join(lines[:mid]),
                "recommendations": "\n".join(lines[mid:]),
                "confidence_score": 0.82,
            }
        except Exception as e:
            logger.error(f"{name} agent error: {e}")
            return {
                "analysis": f"Error during {name} analysis: {e}",
                "recommendations": "Please consult a healthcare professional.",
                "confidence_score": 0.0,
            }
    
    async def analyze_patient(self, patient_data: Dict, image: Optional[Image.Image] = None) -> Dict:
        """Coordinate analysis across multiple agents"""
        results = {}
        
        # Always start with triage
        triage_result = await self.triage_agent.analyze(patient_data)
        results["triage"] = triage_result
        
        # Get general practice assessment
        gp_result = await self.general_practice_agent.analyze(patient_data)
        results["general_practice"] = gp_result
        
        # Add specialist analysis based on symptoms or triage recommendations
        if self._needs_dermatology_consult(patient_data, triage_result):
            if image:
                dermatology_result = await self.dermatology_agent.analyze_with_image(patient_data, image)
            else:
                dermatology_result = await self.dermatology_agent.analyze(patient_data)
            results["dermatology"] = dermatology_result
        
        # Combine results into comprehensive analysis
        comprehensive_analysis = self._combine_agent_results(results, patient_data)
        
        return {
            "comprehensive_analysis": comprehensive_analysis,
            "individual_agents": results,
            "urgency": triage_result.get("urgency", "NON-URGENT"),
            "specialties_recommended": triage_result.get("specialties_recommended", [])
        }
    
    def _needs_dermatology_consult(self, patient_data: Dict, triage_result: Dict) -> bool:
        """Determine if dermatology consultation is needed"""
        symptoms = patient_data.get("symptoms", "").lower()
        skin_keywords = ["rash", "skin", "lesion", "mole", "acne", "eczema", "psoriasis", "dermatitis"]
        
        has_skin_symptoms = any(keyword in symptoms for keyword in skin_keywords)
        dermatology_recommended = "dermatology" in triage_result.get("specialties_recommended", [])
        
        return has_skin_symptoms or dermatology_recommended
    
    def _combine_agent_results(self, results: Dict, patient_data: Dict) -> str:
        """Combine multiple agent analyses into a comprehensive report"""
        patient_name = patient_data.get("name", "Patient")
        
        comprehensive = f"""
# Comprehensive Health Analysis for {patient_name}

## Initial Assessment (Triage)
{results['triage']['analysis']}

**Urgency Level:** {results['triage'].get('urgency', 'NON-URGENT')}

## General Medical Assessment
{results['general_practice']['analysis']}
"""
        
        if 'dermatology' in results:
            comprehensive += f"""
## Dermatological Assessment
{results['dermatology']['analysis']}
"""
        
        if results['triage'].get('specialties_recommended'):
            comprehensive += f"""
## Recommended Specialists
{', '.join(results['triage']['specialties_recommended'])}
"""
        
        comprehensive += """
## Next Steps
Please follow the recommendations above and consult with healthcare professionals as advised.
If you have any follow-up questions about this analysis, feel free to ask for clarification.

---
*This analysis is for educational purposes only and does not replace professional medical consultation.*
"""
        
        return comprehensive
    
    async def handle_followup_question(self, original_analysis: str, question: str, patient_data: Dict) -> str:
        """Handle follow-up questions about the analysis"""
        result = await self.followup_agent.answer_followup(original_analysis, question, patient_data)
        return result.get("response", "I apologize, but I couldn't process your question at this time.")
