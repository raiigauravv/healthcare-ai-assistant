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
from typing import List
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from src.ingestion import ingest_text, ingest_image, ingest_audio
from src.preprocess import preprocess_text
from src.openai_integration import OpenAIHealthcareAssistant
from src.agents import AgentCoordinator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"===== Application Startup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    API_KEY_MISSING = True
else:
    API_KEY_MISSING = False
    logger.info("OpenAI API key found")

try:
    healthcare_ai = OpenAIHealthcareAssistant()
    agent_coordinator = AgentCoordinator()
    INITIALIZATION_SUCCESS = True
    logger.info("AI components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AI components: {e}")
    INITIALIZATION_SUCCESS = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def confidence_bar(score: float) -> str:
    filled = int(round(score * 10))
    bar = "█" * filled + "░" * (10 - filled)
    color = "#22c55e" if score >= 0.7 else "#f59e0b" if score >= 0.4 else "#ef4444"
    return (
        f'<div style="display:flex;align-items:center;gap:10px;margin-top:6px;">'
        f'<span style="font-family:monospace;font-size:1.1rem;color:{color};letter-spacing:2px;">{bar}</span>'
        f'<span style="font-weight:700;color:{color};font-size:0.95rem;">{score:.0%}</span>'
        f'</div>'
    )


def format_agent_html(agent_name: str, agent_data: dict, icon: str, accent: str) -> str:
    if not agent_data:
        return f"<p>❌ Analysis not available for {agent_name}</p>"

    analysis = agent_data.get('analysis', 'No analysis provided')
    recommendations = agent_data.get('recommendations', 'No recommendations provided')
    confidence = agent_data.get('confidence_score', 0.0)
    bar_html = confidence_bar(confidence)

    # Convert newlines to <br> for HTML display
    analysis_html = analysis.replace('\n', '<br>')
    recs_html = recommendations.replace('\n', '<br>')

    return f"""
<div style="font-family:'Inter','Segoe UI',sans-serif;padding:4px 0;">

  <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
    <div style="background:linear-gradient(135deg,{accent}22,{accent}11);border:2px solid {accent}44;
                border-radius:50%;width:52px;height:52px;display:flex;align-items:center;
                justify-content:center;font-size:1.5rem;flex-shrink:0;">{icon}</div>
    <div>
      <h2 style="margin:0;color:#1a202c;font-size:1.25rem;font-weight:700;">{agent_name}</h2>
      <p style="margin:2px 0 0;color:#64748b;font-size:0.8rem;">AI Specialist Analysis</p>
    </div>
    <div style="margin-left:auto;background:{accent}15;border:1px solid {accent}30;
                border-radius:20px;padding:4px 14px;font-size:0.75rem;font-weight:600;color:{accent};">
      AI Analysis
    </div>
  </div>

  <div style="background:#f8faff;border-radius:12px;padding:18px;margin-bottom:16px;
              border-left:4px solid {accent};">
    <h3 style="margin:0 0 12px;color:{accent};font-size:0.9rem;font-weight:700;
               text-transform:uppercase;letter-spacing:0.5px;">🔍 Clinical Assessment</h3>
    <div style="color:#374151;font-size:0.9rem;line-height:1.7;">{analysis_html}</div>
  </div>

  <div style="background:#f0fdf4;border-radius:12px;padding:18px;margin-bottom:16px;
              border-left:4px solid #22c55e;">
    <h3 style="margin:0 0 12px;color:#16a34a;font-size:0.9rem;font-weight:700;
               text-transform:uppercase;letter-spacing:0.5px;">💊 Recommendations</h3>
    <div style="color:#374151;font-size:0.9rem;line-height:1.7;">{recs_html}</div>
  </div>

  <div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:14px 18px;">
    <div style="display:flex;align-items:center;justify-content:space-between;">
      <span style="font-size:0.8rem;font-weight:600;color:#64748b;text-transform:uppercase;
                   letter-spacing:0.5px;">Confidence Score</span>
    </div>
    {bar_html}
  </div>

</div>
"""


def format_summary_html(patient_name, patient_age, patient_gender,
                        has_image, has_audio, gp, cardio, neuro, derma) -> str:
    ts = datetime.now().strftime('%B %d, %Y  %H:%M')

    def mini_card(icon, label, value, color):
        return (
            f'<div style="background:{color}10;border:1px solid {color}25;border-radius:10px;'
            f'padding:12px 16px;display:flex;align-items:center;gap:10px;">'
            f'<span style="font-size:1.3rem;">{icon}</span>'
            f'<div><div style="font-size:0.7rem;color:#64748b;font-weight:600;text-transform:uppercase;">{label}</div>'
            f'<div style="font-size:0.95rem;font-weight:700;color:#1a202c;">{value}</div></div></div>'
        )

    specs = [
        ("🩺 General Physician", gp, "#1a73e8"),
        ("❤️ Cardiologist", cardio, "#ef4444"),
        ("🧠 Neurologist", neuro, "#8b5cf6"),
        ("🔬 Dermatologist", derma, "#0891b2"),
    ]
    spec_rows = ""
    for name, data, color in specs:
        if data:
            conf = data.get('confidence_score', 0)
            bar = confidence_bar(conf)
            spec_rows += f"""
<div style="border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:12px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
    <h4 style="margin:0;color:{color};font-size:0.95rem;font-weight:700;">{name}</h4>
  </div>
  <p style="margin:0 0 8px;color:#374151;font-size:0.85rem;line-height:1.6;">
    {data.get('analysis','')[:300].replace(chr(10),' ')}{'…' if len(data.get('analysis','')) > 300 else ''}
  </p>
  {bar}
</div>"""

    return f"""
<div style="font-family:'Inter','Segoe UI',sans-serif;">

  <div style="background:linear-gradient(135deg,#1a73e8,#0d47a1);border-radius:16px;
              padding:24px;color:white;margin-bottom:20px;">
    <h2 style="margin:0 0 4px;font-size:1.4rem;">🏥 Healthcare Analysis Report</h2>
    <p style="margin:0;opacity:0.85;font-size:0.85rem;">Generated on {ts}</p>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:20px;">
    {mini_card("👤", "Patient", patient_name, "#1a73e8")}
    {mini_card("🎂", "Age", f"{patient_age} years", "#8b5cf6")}
    {mini_card("⚧", "Gender", patient_gender, "#0891b2")}
    {mini_card("📅", "Date", datetime.now().strftime('%b %d, %Y'), "#059669")}
  </div>

  <div style="display:flex;gap:10px;margin-bottom:20px;">
    <div style="flex:1;background:{'#f0fdf4' if has_image else '#fef2f2'};border:1px solid {'#86efac' if has_image else '#fca5a5'};
                border-radius:10px;padding:12px;text-align:center;font-size:0.85rem;font-weight:600;
                color:{'#16a34a' if has_image else '#dc2626'};">
      🖼️ Image {'Uploaded' if has_image else 'Not Provided'}
    </div>
    <div style="flex:1;background:{'#f0fdf4' if has_audio else '#fef2f2'};border:1px solid {'#86efac' if has_audio else '#fca5a5'};
                border-radius:10px;padding:12px;text-align:center;font-size:0.85rem;font-weight:600;
                color:{'#16a34a' if has_audio else '#dc2626'};">
      🎤 Audio {'Uploaded' if has_audio else 'Not Provided'}
    </div>
  </div>

  <h3 style="color:#1a202c;font-size:1rem;font-weight:700;margin:0 0 12px;">
    📋 Specialist Summaries
  </h3>
  {spec_rows}

  <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:12px;padding:16px;margin-top:16px;">
    <p style="margin:0;color:#92400e;font-size:0.82rem;line-height:1.6;">
      ⚠️ <strong>Medical Disclaimer:</strong> This AI analysis is for educational and informational
      purposes only. It does not constitute medical advice, diagnosis, or treatment.
      Always consult a qualified healthcare professional for medical concerns.
    </p>
  </div>

</div>
"""


# ── Business logic ─────────────────────────────────────────────────────────────

def predict_health_issue(text_symptoms, patient_age, patient_gender, patient_name,
                         medical_image=None, audio_file=None):
    ERR = lambda msg: (msg, msg, msg, msg, msg, "")

    if API_KEY_MISSING:
        return ERR("❌ **Configuration Error**: OPENAI_API_KEY not set in HF Space Secrets.")
    if not INITIALIZATION_SUCCESS:
        return ERR("❌ **System Error**: AI components failed to initialize.")
    if not text_symptoms or len(text_symptoms.strip()) < 10:
        return ERR("❌ **Input Error**: Please describe your symptoms in more detail (at least 10 characters).")
    if not patient_name or len(patient_name.strip()) < 2:
        return ERR("❌ **Input Error**: Please provide a valid patient name.")

    logger.info(f"Processing healthcare analysis for patient: {patient_name}")

    text_data   = ingest_text(text_symptoms)
    image_data  = ingest_image(medical_image) if medical_image else None
    audio_data  = ingest_audio(audio_file)    if audio_file    else None
    processed   = preprocess_text(text_data)

    patient_context = {
        "name": patient_name.strip(), "age": patient_age,
        "gender": patient_gender,     "symptoms": processed,
        "has_image": image_data is not None,
        "has_audio": audio_data is not None,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        results     = agent_coordinator.analyze_with_agents(patient_context, processed, image_data, audio_data)
        gp          = results.get('General Physician', {})
        cardio      = results.get('Cardiologist', {})
        neuro       = results.get('Neurologist', {})
        derma       = results.get('Dermatologist', {})

        gp_html     = format_agent_html("General Physician", gp,    "🩺", "#1a73e8")
        cardio_html = format_agent_html("Cardiologist",      cardio, "❤️", "#ef4444")
        neuro_html  = format_agent_html("Neurologist",       neuro,  "🧠", "#8b5cf6")
        derma_html  = format_agent_html("Dermatologist",     derma,  "🔬", "#0891b2")
        summary_html = format_summary_html(
            patient_name, patient_age, patient_gender,
            image_data is not None, audio_data is not None,
            gp, cardio, neuro, derma
        )

        # Plain-text summary for chatbot context
        plain_summary = (
            f"Patient: {patient_name}, {patient_age}yo {patient_gender}\n"
            f"Symptoms: {processed}\n\n"
            f"General Physician: {gp.get('analysis','')} | {gp.get('recommendations','')}\n"
            f"Cardiologist: {cardio.get('analysis','')} | {cardio.get('recommendations','')}\n"
            f"Neurologist: {neuro.get('analysis','')} | {neuro.get('recommendations','')}\n"
            f"Dermatologist: {derma.get('analysis','')} | {derma.get('recommendations','')}\n"
        )

        logger.info(f"Successfully completed analysis for {patient_name}")
        return gp_html, cardio_html, neuro_html, derma_html, summary_html, plain_summary

    except Exception as e:
        logger.error(f"Agent analysis failed: {e}")
        return ERR(f"❌ **AI Analysis Error**: {e}")


def process_followup_question(question, chat_history, patient_name="Patient", analysis_context=""):
    def append(msg):
        chat_history.append([question, msg])
        return "", chat_history

    if API_KEY_MISSING:
        return append("❌ OPENAI_API_KEY not set in HF Space Secrets.")
    if not question or len(question.strip()) < 3:
        return append("❌ Please ask a more detailed question.")

    try:
        parts = []
        if analysis_context and len(analysis_context.strip()) > 50:
            parts.append(
                f"The following specialist analysis was performed for this patient:\n\n"
                f"{analysis_context}\n\n---\n"
                f"Answer the patient's follow-up question specifically based on the above analysis."
            )
        else:
            parts.append(
                "You are a healthcare AI assistant. No prior analysis available — "
                "answer general health questions and recommend professional consultation."
            )
        if chat_history:
            history_text = "\n".join(
                f"Patient: {h[0]}\nAssistant: {h[1]}"
                for h in chat_history[-3:] if h[0] and h[1]
            )
            if history_text:
                parts.append(f"Recent conversation:\n{history_text}")
        parts.append(f"Patient ({patient_name}) asks: {question}")

        response = healthcare_ai.get_health_recommendation("\n\n".join(parts))
        return append(f"**Healthcare AI:** {response}")
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return append("❌ Unable to process your question. Please try again.")


# ── CSS ────────────────────────────────────────────────────────────────────────

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    background: #f0f4ff !important;
    max-width: 1400px !important;
    padding: 0 16px 40px !important;
}

/* ── Cards / blocks ── */
.gr-group, .block, .gr-box {
    background: #ffffff !important;
    border-radius: 16px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
}

/* ── Labels ── */
label > span, .gr-block label, fieldset legend {
    font-weight: 600 !important;
    color: #1e293b !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
}

/* ── Text inputs ── */
input[type="text"], input[type="number"], textarea, select {
    font-family: 'Inter', sans-serif !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    background: #fafbff !important;
    color: #1e293b !important;
}
input:focus, textarea:focus {
    border-color: #1a73e8 !important;
    box-shadow: 0 0 0 3px rgba(26,115,232,0.12) !important;
    outline: none !important;
}

/* ── Primary analyze button ── */
#analyze-btn {
    background: linear-gradient(135deg, #1a73e8 0%, #1557b0 100%) !important;
    border: none !important;
    border-radius: 50px !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.3px !important;
    padding: 14px 0 !important;
    width: 100% !important;
    box-shadow: 0 4px 18px rgba(26,115,232,0.38) !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
    cursor: pointer !important;
}
#analyze-btn:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 28px rgba(26,115,232,0.55) !important;
}
#analyze-btn:active { transform: translateY(-1px) !important; }

/* ── Ask button ── */
#ask-btn {
    background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%) !important;
    border: none !important;
    border-radius: 50px !important;
    color: #fff !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(8,145,178,0.35) !important;
    transition: all 0.25s ease !important;
}
#ask-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(8,145,178,0.55) !important;
}

/* ── Tabs ── */
.tabs .tab-nav {
    background: #f8faff !important;
    border-bottom: 2px solid #e2e8f0 !important;
    gap: 4px !important;
    padding: 8px 8px 0 !important;
    border-radius: 16px 16px 0 0 !important;
}
.tabs .tab-nav button {
    border-radius: 10px 10px 0 0 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    color: #64748b !important;
    border: none !important;
    background: transparent !important;
    padding: 10px 14px !important;
    transition: all 0.2s !important;
}
.tabs .tab-nav button.selected {
    background: #fff !important;
    color: #1a73e8 !important;
    border-bottom: 3px solid #1a73e8 !important;
}
.tabs .tab-nav button:hover:not(.selected) {
    background: #eff6ff !important;
    color: #1a73e8 !important;
}

/* ── Chatbot ── */
#chatbot {
    border: 2px solid #e0ecff !important;
    border-radius: 16px !important;
    background: #f8faff !important;
}
#chatbot .message {
    border-radius: 16px !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
}
#chatbot .message.bot {
    background: linear-gradient(135deg, #eff6ff, #e0f2fe) !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 4px 16px 16px 16px !important;
}
#chatbot .message.user {
    background: linear-gradient(135deg, #1a73e8, #1557b0) !important;
    color: #fff !important;
    border-radius: 16px 4px 16px 16px !important;
}

/* ── Slider ── */
input[type="range"] { accent-color: #1a73e8 !important; }

/* ── Dropdown ── */
.wrap.svelte-1ouvqm4, .wrap-inner { border-radius: 10px !important; }

/* ── Image / Audio upload ── */
.upload-container, .image-frame {
    border: 2px dashed #93c5fd !important;
    border-radius: 14px !important;
    background: #f0f8ff !important;
    transition: all 0.25s !important;
}
.upload-container:hover {
    border-color: #1a73e8 !important;
    background: #eff6ff !important;
}

/* ── Markdown output text ── */
.prose, .md { font-size: 0.88rem !important; line-height: 1.65 !important; color: #374151 !important; }
.prose h2 { color: #1a73e8 !important; font-size: 1.1rem !important; }
.prose h3 { color: #1e293b !important; font-size: 0.95rem !important; }
.prose strong { color: #1e293b !important; }
.prose ul { padding-left: 1.2em !important; }
.prose li { margin-bottom: 4px !important; }

/* ── Examples table ── */
.examples table { border-radius: 12px !important; overflow: hidden !important; }
.examples td, .examples th {
    font-size: 0.82rem !important;
    padding: 8px 12px !important;
}
.examples tr:hover { background: #eff6ff !important; cursor: pointer !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 4px; }
::-webkit-scrollbar-thumb { background: #94a3b8; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #64748b; }
"""


# ── UI ─────────────────────────────────────────────────────────────────────────

HERO_HTML = """
<div style="background:linear-gradient(135deg,#1a73e8 0%,#0d47a1 50%,#1a237e 100%);
            border-radius:20px;padding:36px 40px;color:white;margin-bottom:8px;
            position:relative;overflow:hidden;">

  <!-- Decorative circles -->
  <div style="position:absolute;top:-30px;right:-30px;width:180px;height:180px;
              background:rgba(255,255,255,0.06);border-radius:50%;"></div>
  <div style="position:absolute;bottom:-50px;right:120px;width:120px;height:120px;
              background:rgba(255,255,255,0.04);border-radius:50%;"></div>

  <div style="position:relative;z-index:1;">
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px;">
      <div style="background:rgba(255,255,255,0.15);border-radius:14px;
                  padding:12px;font-size:2rem;backdrop-filter:blur(6px);">🏥</div>
      <div>
        <h1 style="margin:0;font-size:1.9rem;font-weight:800;letter-spacing:-0.5px;">
          Healthcare AI Assistant
        </h1>
        <p style="margin:4px 0 0;opacity:0.8;font-size:0.9rem;font-weight:400;">
          Powered by 4 AI Specialist Agents &nbsp;·&nbsp; Multimodal Analysis
        </p>
      </div>
    </div>

    <div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:16px;">
      <span style="background:rgba(255,255,255,0.15);border-radius:20px;
                   padding:6px 14px;font-size:0.78rem;font-weight:600;backdrop-filter:blur(4px);">
        👨‍⚕️ General Physician
      </span>
      <span style="background:rgba(255,255,255,0.15);border-radius:20px;
                   padding:6px 14px;font-size:0.78rem;font-weight:600;backdrop-filter:blur(4px);">
        ❤️ Cardiologist
      </span>
      <span style="background:rgba(255,255,255,0.15);border-radius:20px;
                   padding:6px 14px;font-size:0.78rem;font-weight:600;backdrop-filter:blur(4px);">
        🧠 Neurologist
      </span>
      <span style="background:rgba(255,255,255,0.15);border-radius:20px;
                   padding:6px 14px;font-size:0.78rem;font-weight:600;backdrop-filter:blur(4px);">
        🔬 Dermatologist
      </span>
      <span style="background:rgba(255,255,255,0.15);border-radius:20px;
                   padding:6px 14px;font-size:0.78rem;font-weight:600;backdrop-filter:blur(4px);">
        💬 AI Chatbot
      </span>
    </div>
  </div>
</div>

<div style="background:linear-gradient(135deg,#fff7ed,#fffbeb);border:1px solid #fed7aa;
            border-radius:12px;padding:12px 18px;margin-bottom:16px;
            display:flex;align-items:center;gap:10px;">
  <span style="font-size:1.2rem;">⚠️</span>
  <span style="font-size:0.82rem;color:#92400e;line-height:1.5;">
    <strong>Medical Disclaimer:</strong> This tool is for educational purposes only and does not
    provide medical diagnoses. Always consult a qualified healthcare professional.
  </span>
</div>
"""

SECTION_INPUT  = '<div style="font-size:1rem;font-weight:700;color:#1e293b;margin:4px 0 12px;display:flex;align-items:center;gap:8px;"><span style="background:#eff6ff;border-radius:8px;padding:4px 10px;color:#1a73e8;">👤 Patient Information</span></div>'
SECTION_SYMP   = '<div style="font-size:1rem;font-weight:700;color:#1e293b;margin:16px 0 12px;display:flex;align-items:center;gap:8px;"><span style="background:#f0fdf4;border-radius:8px;padding:4px 10px;color:#16a34a;">🩺 Describe Symptoms</span></div>'
SECTION_ATTACH = '<div style="font-size:1rem;font-weight:700;color:#1e293b;margin:16px 0 12px;display:flex;align-items:center;gap:8px;"><span style="background:#fdf4ff;border-radius:8px;padding:4px 10px;color:#7c3aed;">📎 Attachments (Optional)</span></div>'
SECTION_RESULT = '<div style="font-size:1rem;font-weight:700;color:#1e293b;margin:4px 0 12px;display:flex;align-items:center;gap:8px;"><span style="background:#eff6ff;border-radius:8px;padding:4px 10px;color:#1a73e8;">📊 AI Specialist Analysis</span></div>'
SECTION_CHAT   = '<div style="font-size:1.05rem;font-weight:700;color:#1e293b;margin:8px 0 14px;display:flex;align-items:center;gap:8px;"><span style="background:#ecfeff;border-radius:8px;padding:4px 12px;color:#0891b2;">💬 AI Health Assistant Chat</span><span style="font-size:0.75rem;color:#64748b;font-weight:400;">Ask follow-up questions based on your analysis</span></div>'

PLACEHOLDER_OUTPUT = """
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
            height:240px;color:#94a3b8;font-family:'Inter',sans-serif;">
  <div style="font-size:3rem;margin-bottom:12px;opacity:0.4;">🔬</div>
  <p style="margin:0;font-size:0.9rem;font-weight:500;">
    Run analysis to see specialist insights here
  </p>
</div>
"""

FOOTER_HTML = """
<div style="background:linear-gradient(135deg,#1e293b,#0f172a);border-radius:16px;
            padding:24px 32px;color:white;margin-top:8px;display:flex;
            flex-wrap:wrap;gap:24px;align-items:center;justify-content:space-between;">
  <div>
    <h3 style="margin:0 0 4px;font-size:1rem;font-weight:700;">🏥 Healthcare AI Assistant</h3>
    <p style="margin:0;opacity:0.6;font-size:0.78rem;">Powered by OpenAI GPT-4o-mini · 4 Specialist Agents</p>
  </div>
  <div style="display:flex;gap:20px;flex-wrap:wrap;">
    <div style="text-align:center;">
      <div style="font-size:1.4rem;font-weight:800;color:#60a5fa;">4</div>
      <div style="font-size:0.7rem;opacity:0.6;text-transform:uppercase;letter-spacing:0.5px;">Specialists</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:1.4rem;font-weight:800;color:#34d399;">3</div>
      <div style="font-size:0.7rem;opacity:0.6;text-transform:uppercase;letter-spacing:0.5px;">Input Modes</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:1.4rem;font-weight:800;color:#f472b6;">∞</div>
      <div style="font-size:0.7rem;opacity:0.6;text-transform:uppercase;letter-spacing:0.5px;">Chat Memory</div>
    </div>
  </div>
  <div style="opacity:0.5;font-size:0.75rem;text-align:right;">
    Version 3.0 · Educational Use Only<br>No medical data is stored
  </div>
</div>
"""


def create_interface():
    with gr.Blocks(css=css, title="🏥 Healthcare AI Assistant") as demo:

        gr.HTML(HERO_HTML)

        with gr.Row(equal_height=False):

            # ── LEFT PANEL — Input ───────────────────────────────────────────
            with gr.Column(scale=2, min_width=320):
                gr.HTML(SECTION_INPUT)
                with gr.Row():
                    patient_name = gr.Textbox(
                        label="Full Name",
                        placeholder="e.g. John Smith",
                        max_lines=1,
                        scale=3,
                    )
                    patient_age = gr.Slider(
                        minimum=1, maximum=120, value=30, step=1,
                        label="Age",
                        scale=2,
                    )
                patient_gender = gr.Dropdown(
                    choices=["Male", "Female", "Non-binary", "Prefer not to say"],
                    value="Male",
                    label="Gender",
                )

                gr.HTML(SECTION_SYMP)
                text_symptoms = gr.Textbox(
                    label="Symptom Description",
                    placeholder=(
                        "Describe your symptoms in detail…\n"
                        "e.g. 'I have had a severe headache for 3 days, with nausea "
                        "and sensitivity to light, especially in the morning.'"
                    ),
                    lines=5,
                    max_lines=12,
                )

                gr.HTML(SECTION_ATTACH)
                with gr.Row():
                    medical_image = gr.Image(
                        label="🖼️ Medical Image",
                        type="filepath",
                        height=140,
                    )
                    audio_file = gr.Audio(
                        label="🎤 Voice Recording",
                        type="filepath",
                    )

                analyze_btn = gr.Button(
                    "🔬  Analyze with AI Specialists",
                    variant="primary",
                    size="lg",
                    elem_id="analyze-btn",
                )

                # Quick examples
                gr.Markdown("##### 💡 Quick Examples")
                gr.Examples(
                    examples=[
                        ["Severe chest pain radiating to my left arm, shortness of breath, sweating — started 2 hours ago during exercise.", 45, "Male",    "John Smith"],
                        ["Persistent headache for a week, blurred vision, numbness in fingers.",                                             32, "Female",  "Sarah Johnson"],
                        ["Dark mole on my back growing and changing colour over the past month, irregular shape, sometimes itches.",          28, "Male",    "Michael Davis"],
                        ["Irregular heartbeat, chest fluttering, dizziness when standing — happening for several days.",                     55, "Female",  "Linda Martinez"],
                    ],
                    inputs=[text_symptoms, patient_age, patient_gender, patient_name],
                    label="",
                )

            # ── RIGHT PANEL — Results ────────────────────────────────────────
            with gr.Column(scale=3, min_width=400):
                gr.HTML(SECTION_RESULT)

                with gr.Tabs():
                    with gr.Tab("🩺 General Physician"):
                        gp_output = gr.HTML(PLACEHOLDER_OUTPUT)
                    with gr.Tab("❤️ Cardiologist"):
                        cardio_output = gr.HTML(PLACEHOLDER_OUTPUT)
                    with gr.Tab("🧠 Neurologist"):
                        neuro_output = gr.HTML(PLACEHOLDER_OUTPUT)
                    with gr.Tab("🔬 Dermatologist"):
                        derma_output = gr.HTML(PLACEHOLDER_OUTPUT)
                    with gr.Tab("📋 Full Summary"):
                        summary_output = gr.HTML(PLACEHOLDER_OUTPUT)

        # ── Hidden state ─────────────────────────────────────────────────────
        analysis_state = gr.State("")

        # ── Chat section ─────────────────────────────────────────────────────
        gr.HTML(SECTION_CHAT)

        chatbot = gr.Chatbot(
            label="",
            height=340,
            show_label=False,
            elem_id="chatbot",
            bubble_full_width=False,
        )

        with gr.Row():
            question_input = gr.Textbox(
                label="",
                placeholder="Ask about your results, medications, lifestyle changes…",
                show_label=False,
                scale=5,
                max_lines=3,
            )
            ask_btn = gr.Button("📤 Ask", variant="secondary", scale=1, elem_id="ask-btn")

        gr.HTML(FOOTER_HTML)

        # ── Event wiring ─────────────────────────────────────────────────────
        analyze_btn.click(
            fn=predict_health_issue,
            inputs=[text_symptoms, patient_age, patient_gender, patient_name, medical_image, audio_file],
            outputs=[gp_output, cardio_output, neuro_output, derma_output, summary_output, analysis_state],
        )
        ask_btn.click(
            fn=process_followup_question,
            inputs=[question_input, chatbot, patient_name, analysis_state],
            outputs=[question_input, chatbot],
        )
        question_input.submit(
            fn=process_followup_question,
            inputs=[question_input, chatbot, patient_name, analysis_state],
            outputs=[question_input, chatbot],
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=False,
    )
