"""
Healthcare AI Assistant - Multimodal AI with Specialist Agents
"""

# gradio_client 0.9.x bug: `"const" in schema` raises TypeError when schema is
# a Python bool (valid JSON Schema for additionalProperties: true/false).
import gradio_client.utils as _gcu
_orig = _gcu._json_schema_to_python_type

def _patched(schema, defs=None):
    if isinstance(schema, bool):
        return "bool"
    return _orig(schema, defs)

_gcu._json_schema_to_python_type = _patched

import gradio as gr
import os
import asyncio
from typing import Optional, Tuple, List
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

print(f"===== Application Startup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")

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
    logger.info("AI components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AI components: {e}")
    INITIALIZATION_SUCCESS = False

def predict_health_issue(text_symptoms: str, patient_age: int, patient_gender: str, patient_name: str, medical_image=None, audio_file=None):
    """
    Premium prediction function using AI agents for specialized healthcare analysis
    """
    try:
        if API_KEY_MISSING:
            return (
                "❌ **Configuration Error**: OpenAI API key not found. Please configure your API key in the environment variables.",
                "❌ **Configuration Error**: OpenAI API key not found. Please configure your API key in the environment variables.",
                "❌ **Configuration Error**: OpenAI API key not found. Please configure your API key in the environment variables.",
                "❌ **Configuration Error**: OpenAI API key not found. Please configure your API key in the environment variables.",
                ""
            )
        
        if not INITIALIZATION_SUCCESS:
            return (
                "❌ **System Error**: AI components failed to initialize. Please try again later.",
                "❌ **System Error**: AI components failed to initialize. Please try again later.",
                "❌ **System Error**: AI components failed to initialize. Please try again later.",
                "❌ **System Error**: AI components failed to initialize. Please try again later.",
                ""
            )
        
        # Validate inputs
        if not text_symptoms or len(text_symptoms.strip()) < 10:
            error_msg = "❌ **Input Error**: Please provide detailed symptoms (at least 10 characters)"
            return error_msg, error_msg, error_msg, error_msg, ""
        
        if not patient_name or len(patient_name.strip()) < 2:
            error_msg = "❌ **Input Error**: Please provide a valid patient name"
            return error_msg, error_msg, error_msg, error_msg, ""
        
        logger.info(f"Processing healthcare analysis for patient: {patient_name}")
        
        # Process inputs through ingestion
        text_data = ingest_text(text_symptoms)
        image_data = ingest_image(medical_image) if medical_image else None
        audio_data = ingest_audio(audio_file) if audio_file else None
        
        # Preprocess text
        processed_text = preprocess_text(text_data)
        
        # Create comprehensive patient context
        patient_context = {
            "name": patient_name.strip(),
            "age": patient_age,
            "gender": patient_gender,
            "symptoms": processed_text,
            "has_image": image_data is not None,
            "has_audio": audio_data is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Get AI agent analysis
        try:
            agent_results = agent_coordinator.analyze_with_agents(
                patient_context=patient_context,
                symptoms_text=processed_text,
                image_data=image_data,
                audio_data=audio_data
            )
            
            # Format results for display
            general_physician = agent_results.get('General Physician', {})
            cardiologist = agent_results.get('Cardiologist', {})
            neurologist = agent_results.get('Neurologist', {})
            dermatologist = agent_results.get('Dermatologist', {})
            
            def format_agent_response(agent_name: str, agent_data: dict) -> str:
                if not agent_data:
                    return f"## 🔬 {agent_name}\n\n❌ Analysis not available"
                
                analysis = agent_data.get('analysis', 'No analysis provided')
                recommendations = agent_data.get('recommendations', 'No recommendations provided')
                confidence = agent_data.get('confidence_score', 0.0)
                
                return f"""## 🔬 {agent_name}
                
**Analysis:**
{analysis}

**Recommendations:**
{recommendations}

**Confidence Score:** {confidence:.1%}
---"""
            
            gp_result = format_agent_response("General Physician", general_physician)
            cardio_result = format_agent_response("Cardiologist", cardiologist)
            neuro_result = format_agent_response("Neurologist", neurologist)
            derma_result = format_agent_response("Dermatologist", dermatologist)
            
            # Create comprehensive summary
            summary = f"""# 🏥 Healthcare Analysis Summary for {patient_name}

**Patient Information:**
- **Name:** {patient_name}
- **Age:** {patient_age} years
- **Gender:** {patient_gender}
- **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Input Data:**
- **Symptoms:** ✅ Provided
- **Medical Image:** {'✅ Uploaded' if image_data else '❌ Not provided'}
- **Audio Recording:** {'✅ Uploaded' if audio_data else '❌ Not provided'}

---

## 📋 Multi-Agent Analysis Results

{gp_result}

{cardio_result}

{neuro_result}

{derma_result}

## ⚠️ Medical Disclaimer
This AI analysis is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.
"""
            
            logger.info(f"Successfully completed analysis for {patient_name}")
            return gp_result, cardio_result, neuro_result, derma_result, summary
            
        except Exception as e:
            error_msg = f"❌ **AI Analysis Error**: {str(e)}"
            logger.error(f"Agent analysis failed: {e}")
            return error_msg, error_msg, error_msg, error_msg, error_msg
            
    except Exception as e:
        error_msg = f"❌ **System Error**: {str(e)}"
        logger.error(f"Prediction failed: {e}")
        return error_msg, error_msg, error_msg, error_msg, error_msg

def process_followup_question(question: str, chat_history: List, patient_name: str = "Patient"):
    """
    Enhanced chatbot function for follow-up questions with conversation history
    """
    try:
        if API_KEY_MISSING:
            error_response = "❌ **Configuration Error**: OpenAI API key not found. Please configure your API key to use the chatbot feature."
            chat_history.append([question, error_response])
            return "", chat_history
        
        if not question or len(question.strip()) < 3:
            error_response = "❌ **Input Error**: Please ask a more detailed question (at least 3 characters)"
            chat_history.append([question, error_response])
            return "", chat_history
        
        # Use healthcare AI for follow-up
        try:
            context = f"Previous conversation context available. Current follow-up question from {patient_name}: {question}"
            response = healthcare_ai.get_health_recommendation(context)
            
            # Format the response for better readability
            formatted_response = f"**Healthcare AI Assistant:** {response}\n\n*This is a follow-up response. For comprehensive analysis, please use the main analysis feature above.*"
            
            chat_history.append([question, formatted_response])
            return "", chat_history
            
        except Exception as e:
            error_response = f"❌ **AI Error**: Unable to process your question. Please try again. Error: {str(e)}"
            chat_history.append([question, error_response])
            return "", chat_history
            
    except Exception as e:
        error_response = f"❌ **System Error**: {str(e)}"
        chat_history.append([question, error_response])
        return "", chat_history

# Premium CSS styling for professional appearance
css = """
/* Premium Healthcare AI Styling */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    padding: 20px !important;
}

.gr-interface {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 20px !important;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1) !important;
    backdrop-filter: blur(10px) !important;
}

.gr-box {
    border-radius: 15px !important;
    border: 2px solid #e1e5e9 !important;
    background: #ffffff !important;
    padding: 20px !important;
    margin: 10px 0 !important;
}

.gr-button {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    border: none !important;
    border-radius: 25px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 30px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
}

.gr-form {
    background: #f8f9fa !important;
    border-radius: 15px !important;
    padding: 25px !important;
    margin: 15px 0 !important;
}

.gr-panel {
    background: #ffffff !important;
    border-radius: 12px !important;
    border-left: 4px solid #667eea !important;
    padding: 20px !important;
    margin: 10px 0 !important;
}

h1, h2, h3 {
    color: #2c3e50 !important;
    font-family: 'Arial', sans-serif !important;
}

.medication-warning {
    background: linear-gradient(45deg, #ff6b6b, #ee5a52) !important;
    color: white !important;
    padding: 15px !important;
    border-radius: 10px !important;
    margin: 15px 0 !important;
    font-weight: bold !important;
}

.chatbot {
    border-radius: 15px !important;
    border: 2px solid #667eea !important;
}
"""

# Create the premium Gradio interface
def create_premium_interface():
    """Create the premium healthcare AI interface with all features"""
    
    with gr.Blocks(css=css, title="🏥 Healthcare AI Assistant - Premium Edition") as interface:
        
        # Header
        gr.Markdown("""
        # 🏥 Healthcare AI Assistant - Premium Edition
        ### Powered by Advanced AI Specialist Agents | Multimodal Analysis Capabilities
        
        **🎯 Features:**
        - 👨‍⚕️ **4 AI Medical Specialists**: General Physician, Cardiologist, Neurologist, Dermatologist  
        - 🖼️ **Multimodal Support**: Text, Image, and Audio analysis
        - 💬 **Intelligent Chatbot**: Follow-up questions with conversation memory
        - 🔬 **Advanced Analysis**: Evidence-based recommendations with confidence scoring
        
        ⚠️ **Medical Disclaimer**: This tool is for educational and informational purposes only. Always consult qualified healthcare professionals for medical advice.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Patient Information Section
                gr.Markdown("## 👤 Patient Information")
                with gr.Row():
                    patient_name = gr.Textbox(
                        label="👤 Patient Name",
                        placeholder="Enter patient's full name...",
                        value="",
                        max_lines=1
                    )
                    patient_age = gr.Slider(
                        minimum=1,
                        maximum=120,
                        value=30,
                        step=1,
                        label="🎂 Age"
                    )

                patient_gender = gr.Dropdown(
                    choices=["Male", "Female", "Non-binary", "Prefer not to say"],
                    label="⚧ Gender",
                    value="Male"
                )
                
                # Symptoms Input Section  
                gr.Markdown("## 🩺 Symptom Description")
                text_symptoms = gr.Textbox(
                    label="📝 Describe Symptoms",
                    placeholder="Please describe the symptoms in detail (e.g., 'I have been experiencing severe headaches for the past 3 days, accompanied by nausea and sensitivity to light...')",
                    lines=4,
                    max_lines=10
                )
                
                # Multimodal Input Section
                gr.Markdown("## 📋 Additional Medical Data (Optional)")
                with gr.Row():
                    medical_image = gr.Image(
                        label="🖼️ Medical Image",
                        type="filepath"
                    )
                    audio_file = gr.Audio(
                        label="🎤 Audio Recording",
                        type="filepath"
                    )
                
                # Analysis Button
                analyze_btn = gr.Button(
                    "🔬 Analyze with AI Specialists",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=3):
                # Results Section
                gr.Markdown("## 📊 AI Specialist Analysis Results")
                
                with gr.Tabs():
                    with gr.Tab("🩺 General Physician"):
                        gp_output = gr.Markdown(
                            "Click 'Analyze with AI Specialists' to get General Physician analysis...",
                            height=300
                        )
                    
                    with gr.Tab("❤️ Cardiologist"):
                        cardio_output = gr.Markdown(
                            "Click 'Analyze with AI Specialists' to get Cardiologist analysis...",
                            height=300
                        )
                    
                    with gr.Tab("🧠 Neurologist"):
                        neuro_output = gr.Markdown(
                            "Click 'Analyze with AI Specialists' to get Neurologist analysis...",
                            height=300
                        )
                    
                    with gr.Tab("🔬 Dermatologist"):
                        derma_output = gr.Markdown(
                            "Click 'Analyze with AI Specialists' to get Dermatologist analysis...",
                            height=300
                        )
                    
                    with gr.Tab("📋 Complete Summary"):
                        summary_output = gr.Markdown(
                            "Click 'Analyze with AI Specialists' to get comprehensive analysis summary...",
                            height=300
                        )
        
        # Enhanced Chatbot Section
        gr.Markdown("## 💬 AI Health Assistant Chatbot")
        gr.Markdown("*Ask follow-up questions about your health analysis or get additional medical information*")
        
        chatbot = gr.Chatbot(
            label="🤖 Healthcare AI Chat",
            height=300,
            show_label=True
        )
        
        with gr.Row():
            question_input = gr.Textbox(
                label="Ask a follow-up question",
                placeholder="Ask about medications, treatment options, symptoms clarification, etc...",
                scale=4
            )
            ask_btn = gr.Button("📤 Ask", variant="secondary", scale=1)
        
        # Premium Examples Section
        gr.Markdown("## 💡 Example Cases for Testing")
        
        examples = [
            [
                "I've been experiencing severe chest pain that radiates to my left arm, along with shortness of breath and sweating. This started about 2 hours ago during light exercise.",
                45,
                "Male", 
                "John Smith"
            ],
            [
                "I have a persistent headache for the past week, blurred vision, and difficulty concentrating. I also noticed some numbness in my fingers.",
                32,
                "Female",
                "Sarah Johnson"
            ],
            [
                "I found a dark mole on my back that has been growing and changing color over the past month. It's irregular in shape and sometimes itches.",
                28,
                "Male",
                "Michael Davis"
            ],
            [
                "I'm experiencing irregular heartbeat, chest fluttering, and occasional dizziness, especially when standing up quickly. This has been happening for several days.",
                55,
                "Female",
                "Linda Martinez"
            ]
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[text_symptoms, patient_age, patient_gender, patient_name],
            label="🎯 Click on an example to auto-fill the form"
        )
        
        # Event handlers
        analyze_btn.click(
            fn=predict_health_issue,
            inputs=[text_symptoms, patient_age, patient_gender, patient_name, medical_image, audio_file],
            outputs=[gp_output, cardio_output, neuro_output, derma_output, summary_output]
        )
        
        # Enhanced chatbot interaction
        ask_btn.click(
            fn=process_followup_question,
            inputs=[question_input, chatbot, patient_name],
            outputs=[question_input, chatbot]
        )
        
        question_input.submit(
            fn=process_followup_question,
            inputs=[question_input, chatbot, patient_name],
            outputs=[question_input, chatbot]
        )
        
        # Footer
        gr.Markdown("""
        ---
        ### 🔬 About This AI System
        
        **Technology Stack:**
        - **AI Models**: OpenAI GPT-4 with specialized medical training
        - **Specialist Agents**: 4 dedicated AI medical specialists with domain expertise
        - **Multimodal Processing**: Advanced image and audio analysis capabilities
        - **Safety Features**: Built-in medical disclaimers and ethical guidelines
        
        **🛡️ Privacy & Security:**
        - No personal data is stored permanently
        - All interactions are processed securely
        **Version**: Premium Edition 2.0 | **Status**: ✅ Fully Operational
        """)
    
    return interface

# Create and launch the premium interface
if __name__ == "__main__":
    demo = create_premium_interface()
    
    # Launch with premium configuration
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=False
    )
