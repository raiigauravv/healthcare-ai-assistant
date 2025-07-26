import gradio as gr
import os
import logging
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check API key availability
API_KEY_MISSING = not bool(os.getenv("OPENAI_API_KEY"))
INITIALIZATION_SUCCESS = not API_KEY_MISSING

if not API_KEY_MISSING:
    try:
        # Import only if API key is available
        from src.ingestion import ingest_text, ingest_image, ingest_audio
        from src.preprocess import preprocess_text
        from src.agents import AgentCoordinator
        from src.openai_integration import HealthcareAI
        
        # Initialize components
        healthcare_ai = HealthcareAI()
        agent_coordinator = AgentCoordinator()
        
        logger.info("Healthcare AI system initialized successfully")
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        INITIALIZATION_SUCCESS = False
        API_KEY_MISSING = True

def process_health_query(patient_name, symptoms, age, gender):
    """Simple health analysis function"""
    if API_KEY_MISSING or not INITIALIZATION_SUCCESS:
        return (
            "⚠️ **Configuration Required**\n\nTo use this app, please:\n1. Go to HF Space settings\n2. Add secret: OPENAI_API_KEY\n3. Restart the space",
            "Please configure your OpenAI API key first.",
            0.0
        )
    
    if not symptoms or len(symptoms.strip()) < 10:
        return "❌ Please provide detailed symptoms (at least 10 characters)", "", 0.0
    
    try:
        # Process input
        processed_text = ingest_text(symptoms)
        cleaned_text = preprocess_text(processed_text)
        
        # Prepare patient data
        patient_data = {
            "name": patient_name.strip() if patient_name.strip() else "Patient",
            "age": str(age),
            "gender": gender,
            "symptoms": cleaned_text,
            "has_image": False,
            "has_audio": False,
            "audio_analysis": ""
        }
        
        # Get analysis
        result = asyncio.run(agent_coordinator.analyze_patient(patient_data, None))
        
        analysis = result.get("comprehensive_analysis", "Analysis unavailable")
        urgency = result.get("urgency", "NON-URGENT")
        
        # Generate recommendations
        if urgency == "EMERGENCY":
            recommendations = "🚨 **EMERGENCY** - Call 911 immediately!"
        elif urgency == "URGENT":
            recommendations = "🔴 **Seek medical care today** - Contact your doctor or urgent care"
        elif urgency == "SEMI-URGENT":
            recommendations = "🟡 **Schedule appointment** - See doctor within 1-3 days"
        else:
            recommendations = "🟢 **Monitor symptoms** - Routine care if needed"
        
        confidence = result.get("confidence", 0.8)
        
        return analysis, recommendations, confidence
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return f"❌ Error: {str(e)}", "Please try again", 0.0

def handle_followup(question, analysis, age, gender, name, history):
    """Simple follow-up handler"""
    if API_KEY_MISSING or not INITIALIZATION_SUCCESS:
        response = "⚠️ Please configure your OpenAI API key first."
        new_history = history + f"\n\n**You**: {question}\n**Agent**: {response}"
        return new_history, ""
    
    if not question.strip():
        return history, ""
    
    try:
        # Simple follow-up response using healthcare AI
        patient_data = {
            "name": name or "Patient",
            "age": str(age),
            "gender": gender,
            "symptoms": "Follow-up question"
        }
        
        response = asyncio.run(
            agent_coordinator.handle_followup_question(analysis, question, patient_data)
        )
        
        # Format conversation
        new_history = history + f"\n\n**You**: {question}\n**Follow-up Agent**: {response}\n---"
        
        return new_history, ""
        
    except Exception as e:
        response = f"Error: {str(e)}"
        new_history = history + f"\n\n**You**: {question}\n**Agent**: {response}"
        return new_history, ""

# Create the interface
def create_simple_interface():
    with gr.Blocks(title="Healthcare AI Assistant") as demo:
        gr.Markdown("# 🏥 Healthcare AI Assistant")
        gr.Markdown("*AI-powered health analysis with multiple specialist agents*")
        
        # Status
        if API_KEY_MISSING:
            gr.Markdown("⚠️ **Status**: Configuration required - Add OpenAI API key")
        else:
            gr.Markdown("✅ **Status**: Online and ready")
        
        with gr.Row():
            with gr.Column():
                patient_name = gr.Textbox(label="Patient Name", placeholder="Enter your name")
                symptoms = gr.Textbox(
                    label="Describe Your Symptoms", 
                    placeholder="Tell me about your symptoms in detail...", 
                    lines=4
                )
                with gr.Row():
                    age = gr.Number(label="Age", value=30, minimum=1, maximum=120)
                    gender = gr.Dropdown(
                        label="Gender",
                        choices=["Male", "Female", "Other", "Prefer not to say"],
                        value="Prefer not to say"
                    )
                
                analyze_btn = gr.Button("🔍 Analyze Symptoms", variant="primary")
                
            with gr.Column():
                analysis_output = gr.Textbox(label="🩺 Analysis", lines=10, interactive=False)
                recommendations_output = gr.Textbox(label="💡 Recommendations", lines=4, interactive=False)
                confidence_output = gr.Number(label="🎯 Confidence", interactive=False)
        
        # Follow-up section
        gr.Markdown("## 💬 Follow-up Questions")
        followup_history = gr.Textbox(
            label="Conversation", 
            lines=6, 
            interactive=False,
            placeholder="Your conversation will appear here..."
        )
        
        with gr.Row():
            followup_question = gr.Textbox(
                label="Ask a follow-up question",
                placeholder="e.g., What can I do at home?",
                scale=4
            )
            followup_btn = gr.Button("Send", variant="primary", scale=1)
        
        clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        # Store states
        analysis_state = gr.State("")
        
        # Event handlers
        analyze_btn.click(
            fn=process_health_query,
            inputs=[patient_name, symptoms, age, gender],
            outputs=[analysis_output, recommendations_output, confidence_output]
        ).then(
            fn=lambda x: x,
            inputs=[analysis_output],
            outputs=[analysis_state]
        )
        
        followup_btn.click(
            fn=handle_followup,
            inputs=[followup_question, analysis_state, age, gender, patient_name, followup_history],
            outputs=[followup_history, followup_question]
        )
        
        followup_question.submit(
            fn=handle_followup,
            inputs=[followup_question, analysis_state, age, gender, patient_name, followup_history],
            outputs=[followup_history, followup_question]
        )
        
        clear_btn.click(
            fn=lambda: "",
            outputs=[followup_history]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["Sarah", "Severe chest pain and difficulty breathing", 45, "Female"],
                ["Mike", "Persistent cough with fever for 3 weeks", 35, "Male"],
                ["Alex", "Red itchy rash on face for several days", 28, "Other"]
            ],
            inputs=[patient_name, symptoms, age, gender]
        )
    
    return demo

# Launch
if __name__ == "__main__":
    demo = create_simple_interface()
    
    # Simple launch logic
    is_hf_space = os.environ.get('SPACE_ID') is not None
    
    if is_hf_space:
        demo.launch()
    else:
        demo.launch(share=True)
