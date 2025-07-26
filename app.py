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

# Check for OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    API_KEY_MISSING = True
else:
    API_KEY_MISSING = False
    logger.info("OpenAI API key found")

# Initialize OpenAI assistant and Agent Coordinator
try:
    healthcare_ai = OpenAIHealthcareAssistant()
    agent_coordinator = AgentCoordinator()
    INITIALIZATION_SUCCESS = True
except Exception as e:
    logger.error(f"Failed to initialize AI components: {e}")
    INITIALIZATION_SUCCESS = False

def predict_health_issue(text_symptoms: str, patient_age: int, patient_gender: str, patient_name: str, medical_image=None, audio_file=None):
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
        # Check if API key is available
        if API_KEY_MISSING or not INITIALIZATION_SUCCESS:
            error_msg = """
❌ **Configuration Error**: OpenAI API key not found.

🔧 **For Hugging Face Spaces Administrators**:
1. Go to your Space settings
2. Add a new secret: `OPENAI_API_KEY`
3. Paste your OpenAI API key as the value
4. Restart the Space

🏠 **For Local Development**:
Add your OpenAI API key to the .env file.

This is a demonstration of a healthcare AI system that requires OpenAI API access to function.
"""
            return error_msg, "Please configure the OpenAI API key to use this application.", 0.0
        
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
            "name": patient_name.strip() if patient_name.strip() else "Patient",  # Use provided name or default
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


def multimodal_predict(text_symptoms: str, medical_image=None, audio_file=None, patient_age: int = 30, patient_gender: str = "Not specified", patient_name: str = ""):
    """Wrapper function for compatibility with the interface"""
    return predict_health_issue(text_symptoms, patient_age, patient_gender, patient_name, medical_image, audio_file)


def handle_followup_question(question: str, original_analysis: str, patient_age: int, patient_gender: str, patient_name: str, chat_history: list, current_history: str) -> tuple:
    """
    Handle follow-up questions about the medical analysis in simple text format
    """
    try:
        if not question or not question.strip():
            return current_history, "", chat_history
        
        # Check if API key is available
        if API_KEY_MISSING or not INITIALIZATION_SUCCESS:
            error_msg = "⚠️ OpenAI API key not configured. Please configure the API key first."
            new_history = current_history + f"\n\n👤 You: {question}\n🤖 Agent: {error_msg}"
            return new_history, "", chat_history
        
        if not original_analysis or "Error" in original_analysis:
            error_msg = "Please complete a health analysis first before asking follow-up questions."
            new_history = current_history + f"\n\n👤 You: {question}\n🤖 Agent: {error_msg}"
            return new_history, "", chat_history
        
        # Prepare patient data for follow-up
        patient_data = {
            "name": patient_name.strip() if patient_name.strip() else "Patient",
            "age": str(patient_age),
            "gender": patient_gender,
            "symptoms": "Previous analysis completed"
        }
        
        # Use the follow-up agent
        response = asyncio.run(
            agent_coordinator.handle_followup_question(original_analysis, question, patient_data)
        )
        
        # Clean up the response to remove placeholder names and make it more conversational
        response = response.replace("[Your Name]", "Healthcare AI Assistant")
        response = response.replace("Dear " + patient_data["name"] + ",", f"Hello {patient_data['name']},")
        response = response.replace("Best,\n[Your Name]", "Hope this helps!\n\n- Your Healthcare AI Assistant 🤖")
        
        # Add to chat history
        chat_history.append((question, response))
        
        # Update display history
        new_history = current_history + f"\n\n👤 You: {question}\n\n🤖 Agent: {response}"
        
        return new_history, "", chat_history
        
    except Exception as e:
        logger.error(f"Follow-up question error: {e}")
        error_msg = f"Sorry, I encountered an error processing your question: {str(e)}"
        new_history = current_history + f"\n\n👤 You: {question}\n🤖 Agent: {error_msg}"
        return new_history, "", chat_history

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
                
                # Patient name input
                patient_name = gr.Textbox(
                    label="Patient Name",
                    placeholder="Enter your name (optional)",
                    value="",
                    max_lines=1
                )
                
                # Text input
                text_symptoms = gr.Textbox(
                    label="Describe Symptoms",
                    placeholder="Please describe your symptoms in detail (e.g., 'I have been experiencing chest pain and shortness of breath for the past 2 days...')",
                    lines=4,
                    max_lines=8
                )
                
                with gr.Row():
                    patient_age = gr.Number(
                        minimum=0,
                        maximum=120,
                        value=30,
                        step=1,
                        label="Age",
                        precision=0
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
                    "🔍 Analyze with Agents", 
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
                status_color = "#28a745" if not API_KEY_MISSING and INITIALIZATION_SUCCESS else "#dc3545"
                status_text = "Online and Ready" if not API_KEY_MISSING and INITIALIZATION_SUCCESS else "Configuration Required"
                
                gr.HTML(f"""
                <div style="margin-top: 20px; padding: 15px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <strong>🤖 Powered by:</strong> OpenAI GPT-4, CLIP, and Whisper models<br>
                    <strong>🧠 AI Agents:</strong> Triage, Dermatology, General Practice, Follow-up<br>
                    <strong>⚡ Status:</strong> <span style="color: {status_color}; font-weight: bold;">{status_text}</span>
                </div>
                """)
        
        # Follow-up Questions Section - Simple Chat Style
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 💬 Follow-up Questions")
                gr.Markdown("*Ask follow-up questions about your health analysis for additional guidance.*")
                
                # Simple chat interface
                followup_history = gr.Textbox(
                    label="Conversation History",
                    lines=8,
                    max_lines=12,
                    interactive=False,
                    placeholder="Your conversation with the healthcare agent will appear here..."
                )
                
                with gr.Row():
                    followup_question = gr.Textbox(
                        label="",
                        placeholder="Ask me anything about your health analysis...",
                        lines=1,
                        scale=4
                    )
                    followup_btn = gr.Button("Send", variant="primary", scale=1)
                
                # Clear chat button
                clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")
        
        # Store the analysis and chat history for follow-up questions
        analysis_state = gr.State(value="")
        chat_history = gr.State(value=[])
        
        # Example inputs
        gr.Markdown("## 💡 Example Inputs")
        
        # Example cases
        examples = [
            [
                "Sarah Johnson",
                "I have severe chest pain, difficulty breathing, and I'm sweating. This started 30 minutes ago.",
                45,
                "Female",
                None,
                None
            ],
            [
                "Mike Smith",
                "I've had a persistent cough for 3 weeks with yellow-green phlegm, fever, and difficulty breathing. I'm also feeling very tired.",
                45,
                "Male",
                None,
                None
            ],
            [
                "Alex Taylor",
                "I noticed a small, dark mole on my arm that has changed color and size over the past month. It's also slightly raised and sometimes itches.",
                35,
                "Other",
                None,
                None
            ]
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[patient_name, text_symptoms, patient_age, patient_gender, medical_image, audio_file],
            label="Click on an example to try it out"
        )
        
        # Connect the interface
        submit_btn.click(
            fn=multimodal_predict,
            inputs=[text_symptoms, medical_image, audio_file, patient_age, patient_gender, patient_name],
            outputs=[analysis_output, recommendations_output, confidence_output],
            show_progress=True
        ).then(
            fn=lambda analysis, *args: analysis,  # Store analysis for follow-up
            inputs=[analysis_output, recommendations_output, confidence_output],
            outputs=[analysis_state]
        )
        
        # Connect follow-up question handler (simple text style)
        followup_btn.click(
            fn=handle_followup_question,
            inputs=[followup_question, analysis_state, patient_age, patient_gender, patient_name, chat_history, followup_history],
            outputs=[followup_history, followup_question, chat_history],
            show_progress=True
        )
        
        # Enter key support for followup
        followup_question.submit(
            fn=handle_followup_question,
            inputs=[followup_question, analysis_state, patient_age, patient_gender, patient_name, chat_history, followup_history],
            outputs=[followup_history, followup_question, chat_history],
            show_progress=True
        )
        
        # Clear chat functionality
        clear_chat_btn.click(
            fn=lambda: ("", []),
            outputs=[followup_history, chat_history]
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
    demo.launch(share=True)
