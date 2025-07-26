"""
Healthcare AI Assistant - Enhanced with AI Specialist Agents
Multimodal healthcare AI with OpenAI integration and specialist agent system
"""

import gradio as gr
import os
import asyncio
from typing import Optional, Tuple
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from src.ingestion import ingest_text, ingest_image, ingest_audio
from src.preprocess import preprocess_text
from src.openai_integration import OpenAIHealthcareAssistant
from src.agents import AgentCoordinator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI assistant and Agent Coordinator
healthcare_ai = OpenAIHealthcareAssistant()
agent_coordinator = AgentCoordinator()

def multimodal_predict(
    text_symptoms: str,
    medical_image: Optional[str] = None,
    audio_file: Optional[str] = None,
    patient_age: int = 30,
    patient_gender: str = "Not specified"
) -> Tuple[str, str, float]:
    """
    Enhanced prediction function using AI agents for specialized healthcare analysis
    
    Args:
        text_symptoms: Patient's symptom description
        medical_image: Uploaded medical image (optional)
        audio_file: Uploaded audio recording (optional)
        patient_age: Patient's age
        patient_gender: Patient's gender
        
    Returns:
        Tuple of (agent_analysis, recommendations, confidence_score)
    """
    try:
        logger.info("Starting multimodal healthcare prediction with AI agents")
        
        # Validate inputs
        if not text_symptoms or len(text_symptoms.strip()) < 10:
            return "❌ Error: Please provide detailed symptoms (at least 10 characters)", "", 0.0
        
        # Process text input
        processed_text = ingest_text(text_symptoms)
        cleaned_text = preprocess_text(processed_text)
        
        # Process image if provided
        processed_image = None
        if medical_image:
            try:
                processed_image = ingest_image(medical_image)
                logger.info("Medical image processed successfully")
            except Exception as e:
                logger.error(f"Image processing error: {e}")
        
        # Process audio if provided (enhanced for future audio analysis)
        audio_analysis = ""
        if audio_file:
            try:
                audio_data = ingest_audio(audio_file)
                audio_analysis = healthcare_ai.analyze_audio_symptoms(audio_file, text_symptoms)
                logger.info("Audio processed successfully")
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                audio_analysis = "⚠️ Audio analysis unavailable"
        
        # Prepare patient data for agents
        patient_data = {
            "name": "Patient",  # Default name since we don't have name input currently
            "age": str(patient_age),
            "gender": patient_gender,
            "symptoms": cleaned_text,
            "has_image": processed_image is not None,
            "has_audio": bool(audio_file),
            "audio_analysis": audio_analysis
        }
        
        # Use Agent Coordinator for comprehensive analysis
        agent_results = asyncio.run(agent_coordinator.analyze_patient(patient_data, processed_image))
        
        # Extract comprehensive analysis
        comprehensive_analysis = agent_results.get("comprehensive_analysis", "Analysis unavailable")
        urgency_level = agent_results.get("urgency", "NON-URGENT")
        specialties = agent_results.get("specialties_recommended", [])
        
        # Generate recommendations based on urgency and specialties
        recommendations = f"""
## 🚨 Urgency Level: {urgency_level}

## 🏥 Recommended Next Steps:
"""
        
        if urgency_level == "EMERGENCY":
            recommendations += """
⚠️ **SEEK IMMEDIATE EMERGENCY CARE**
- Call 911 or go to the nearest emergency room immediately
- Do not delay medical attention
"""
        elif urgency_level == "URGENT":
            recommendations += """
🔴 **Seek medical care today**
- Contact your doctor or visit urgent care within a few hours
- Monitor symptoms closely
"""
        elif urgency_level == "SEMI-URGENT":
            recommendations += """
🟡 **Schedule medical appointment**
- See your healthcare provider within 1-3 days
- Continue monitoring symptoms
"""
        else:
            recommendations += """
🟢 **Routine medical care**
- Schedule regular appointment with your healthcare provider
- Continue self-care measures as appropriate
"""
        
        if specialties:
            recommendations += f"\n## 👩‍⚕️ Specialist Consultations Recommended:\n"
            for specialty in specialties:
                recommendations += f"- {specialty.title()}\n"
        
        recommendations += """
## 📞 Important Reminders:
- This analysis is for educational purposes only
- Always consult healthcare professionals for medical decisions
- Trust your instincts - if you feel something is wrong, seek medical care
"""
        
        # Calculate confidence based on agent consensus and data quality
        confidence = 0.8 if processed_image else 0.7
        if urgency_level in ["EMERGENCY", "URGENT"]:
            confidence = max(confidence, 0.85)  # Higher confidence for urgent cases
        
        logger.info(f"Agent-based prediction completed - Urgency: {urgency_level}")
        
        return comprehensive_analysis, recommendations, confidence
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return f"❌ Error during analysis: {str(e)}", "Please try again or consult a healthcare professional", 0.0


def handle_followup_question(question: str, original_analysis: str, patient_age: int, patient_gender: str) -> str:
    """
    Handle follow-up questions about the medical analysis
    """
    try:
        if not question or not question.strip():
            return "Please ask a specific question about your health analysis."
        
        if not original_analysis or "Error" in original_analysis:
            return "Please complete a health analysis first before asking follow-up questions."
        
        # Prepare patient data for follow-up
        patient_data = {
            "name": "Patient",
            "age": str(patient_age),
            "gender": patient_gender,
            "symptoms": "Previous analysis completed"
        }
        
        # Use the follow-up agent
        response = asyncio.run(
            agent_coordinator.handle_followup_question(original_analysis, question, patient_data)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Follow-up question error: {e}")
        return f"Sorry, I encountered an error processing your question: {str(e)}"

def create_demo_interface():
    """Create and configure the Gradio interface"""
    
    # Custom CSS for healthcare theme
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .warning {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: #000000;
        font-weight: 500;
    }
    """
    
    with gr.Blocks(css=css, title="Healthcare AI Assistant") as interface:
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🏥 Healthcare AI Assistant</h1>
            <p>Advanced multimodal AI with specialist agents for healthcare analysis</p>
            <p style="font-size: 0.9em;">🤖 Featuring: Triage • Dermatology • General Practice • Follow-up Agents</p>
        </div>
        """)
        
        # Disclaimer
        gr.HTML("""
        <div class="warning">
            ⚠️ <strong>Medical Disclaimer:</strong> This is a demonstration tool for educational purposes only. 
            It does not provide medical advice and should not be used as a substitute for professional medical consultation, 
            diagnosis, or treatment. Always seek advice from qualified healthcare professionals.
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📝 Input Information")
                
                # Text input
                text_symptoms = gr.Textbox(
                    label="Describe Symptoms",
                    placeholder="Please describe your symptoms in detail (e.g., 'I have been experiencing chest pain and shortness of breath for the past 2 days...')",
                    lines=4,
                    max_lines=8
                )
                
                with gr.Row():
                    patient_age = gr.Slider(
                        minimum=0,
                        maximum=120,
                        value=30,
                        step=1,
                        label="Age"
                    )
                    patient_gender = gr.Dropdown(
                        choices=["Male", "Female", "Other", "Not specified"],
                        value="Not specified",
                        label="Gender"
                    )
                
                # File uploads
                medical_image = gr.Image(
                    label="Medical Image (Optional)",
                    type="filepath",
                    height=200
                )
                
                audio_file = gr.Audio(
                    label="Audio Recording (Optional)",
                    type="filepath"
                )
                
                # Submit button
                submit_btn = gr.Button(
                    "🔍 Analyze Health Data", 
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("## 📊 Analysis Results")
                
                # Output displays
                analysis_output = gr.Textbox(
                    label="🔬 Health Analysis",
                    lines=8,
                    max_lines=12,
                    interactive=False
                )
                
                recommendations_output = gr.Textbox(
                    label="💡 Recommendations",
                    lines=6,
                    max_lines=10,
                    interactive=False
                )
                
                confidence_output = gr.Slider(
                    label="🎯 Confidence Score",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    interactive=False,
                    show_label=True
                )
                
                # Status indicator
                gr.HTML("""
                <div style="margin-top: 20px; padding: 15px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <strong>🤖 Powered by:</strong> OpenAI GPT-4, CLIP, and Whisper models<br>
                    <strong>🧠 AI Agents:</strong> Triage, Dermatology, General Practice, Follow-up<br>
                    <strong>⚡ Status:</strong> <span style="color: #28a745; font-weight: bold;">Online and Ready</span>
                </div>
                """)
        
        # Follow-up Questions Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 💬 Follow-up Questions")
                gr.Markdown("*Have questions about your analysis? Ask our follow-up agent for clarification.*")
                
                followup_question = gr.Textbox(
                    label="Ask a follow-up question",
                    placeholder="e.g., 'What should I do if the pain gets worse?' or 'Are there any home remedies I can try?'",
                    lines=2
                )
                
                followup_btn = gr.Button("Ask Follow-up Question", variant="secondary")
                
                followup_response = gr.Textbox(
                    label="Follow-up Response",
                    lines=4,
                    interactive=False
                )
        
        # Store the analysis for follow-up questions
        analysis_state = gr.State(value="")
        
        # Example inputs
        gr.Markdown("## 💡 Example Inputs")
        
        examples = [
            [
                "I have been experiencing persistent headaches for the past week, along with nausea and sensitivity to light. The pain is usually on one side of my head and gets worse with physical activity.",
                None,
                None,
                "28",
                "Female"
            ],
            [
                "I've had a persistent cough for 3 weeks with yellow-green phlegm, fever, and difficulty breathing. I'm also feeling very tired.",
                None,
                None,
                "45",
                "Male"
            ],
            [
                "I noticed a small, dark mole on my arm that has changed color and size over the past month. It's also slightly raised and sometimes itches.",
                None,
                None,
                "35",
                "Other"
            ]
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[text_symptoms, medical_image, audio_file, patient_age, patient_gender],
            label="Click on an example to try it out"
        )
        
        # Connect the interface
        submit_btn.click(
            fn=multimodal_predict,
            inputs=[text_symptoms, medical_image, audio_file, patient_age, patient_gender],
            outputs=[analysis_output, recommendations_output, confidence_output],
            show_progress=True
        ).then(
            fn=lambda analysis, *args: analysis,  # Store analysis for follow-up
            inputs=[analysis_output, recommendations_output, confidence_output],
            outputs=[analysis_state]
        )
        
        # Connect follow-up question handler
        followup_btn.click(
            fn=handle_followup_question,
            inputs=[followup_question, analysis_state, patient_age, patient_gender],
            outputs=[followup_response],
            show_progress=True
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <p><strong>Healthcare AI Assistant</strong> - Demonstrating multimodal AI in healthcare</p>
            <p>Built with 💝 using OpenAI, Gradio, and Hugging Face Spaces</p>
            <p style="font-size: 0.8em; color: #666;">
                Remember: This is for demonstration purposes only. Always consult healthcare professionals for medical advice.
            </p>
        </div>
        """)
    
    return interface

# Launch the interface
if __name__ == "__main__":
    demo = create_demo_interface()
    # For Hugging Face Spaces deployment
    demo.launch()
