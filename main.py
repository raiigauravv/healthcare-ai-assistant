"""
Healthcare AI Assistant — FastAPI backend
Serves React SPA on / and AI endpoints on /api/*
"""
import os
import tempfile
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from src.ingestion import ingest_text, ingest_image, ingest_audio
from src.preprocess import preprocess_text
from src.openai_integration import OpenAIHealthcareAssistant
from src.agents import AgentCoordinator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"===== Startup {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")

try:
    healthcare_ai = OpenAIHealthcareAssistant()
    agent_coordinator = AgentCoordinator()
    READY = True
    logger.info("AI components initialised")
except Exception as exc:
    logger.error(f"AI init failed: {exc}")
    READY = False

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Healthcare AI Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Models ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    history: List = []
    patient_name: str = "Patient"
    analysis_context: str = ""


# ── API routes ───────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "ai_ready": READY,
        "api_configured": READY and not healthcare_ai.mock_mode,
    }


@app.post("/api/analyze")
async def analyze(
    symptoms:       str            = Form(...),
    patient_name:   str            = Form(...),
    patient_age:    int            = Form(30),
    patient_gender: str            = Form("Not specified"),
    image:          Optional[UploadFile] = File(None),
    audio:          Optional[UploadFile] = File(None),
):
    if not READY:
        raise HTTPException(503, "AI not initialised")
    if len(symptoms.strip()) < 10:
        raise HTTPException(400, "Symptoms too short (min 10 chars)")
    if len(patient_name.strip()) < 2:
        raise HTTPException(400, "Patient name required")

    image_data = audio_data = None
    tmp_files: list[str] = []

    try:
        if image and image.filename:
            suffix = os.path.splitext(image.filename)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(await image.read())
                tmp_files.append(f.name)
                image_data = ingest_image(f.name)

        if audio and audio.filename:
            suffix = os.path.splitext(audio.filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(await audio.read())
                tmp_files.append(f.name)
                audio_data = ingest_audio(f.name)

        processed = preprocess_text(ingest_text(symptoms))

        # ── Whisper audio transcription ──────────────────────────────────────
        audio_transcription = ""
        if audio_data is not None and tmp_files:
            # The last temp file written was the audio file
            audio_path = next((p for p in reversed(tmp_files)
                               if any(p.endswith(x) for x in ('.wav','.mp3','.m4a','.ogg','.flac','.webm'))), None)
            if audio_path:
                audio_transcription = healthcare_ai.analyze_audio_symptoms(audio_path, processed)
                logger.info(f"Whisper result: {audio_transcription[:80]}")

        # Enrich the symptoms text with any voice transcription
        full_symptoms = processed
        if audio_transcription and "transcription" in audio_transcription.lower():
            full_symptoms += f"\n\nVoice note: {audio_transcription}"

        ctx = {
            "name": patient_name.strip(), "age": patient_age,
            "gender": patient_gender,     "symptoms": full_symptoms,
            "has_image": image_data is not None,
            "has_audio": audio_data is not None,
        }

        # image_data (PIL Image) is passed to each specialist who will see it via GPT-4o Vision
        results = agent_coordinator.analyze_with_agents(ctx, full_symptoms, image_data, audio_data)
        gp     = results.get("General Physician", {})
        cardio = results.get("Cardiologist", {})
        neuro  = results.get("Neurologist", {})
        derma  = results.get("Dermatologist", {})

        plain = (
            f"Patient: {patient_name}, {patient_age}yo {patient_gender}\n"
            f"Symptoms: {full_symptoms}\n"
            + (f"Image: uploaded and examined by GPT-4o Vision\n" if image_data else "")
            + (f"Audio: {audio_transcription}\n" if audio_transcription else "")
            + f"\nGP: {gp.get('analysis','')} | {gp.get('recommendations','')}\n"
            f"Cardio: {cardio.get('analysis','')} | {cardio.get('recommendations','')}\n"
            f"Neuro: {neuro.get('analysis','')} | {neuro.get('recommendations','')}\n"
            f"Derm: {derma.get('analysis','')} | {derma.get('recommendations','')}\n"
        )

        return {
            "success": True,
            "general_physician": gp, "cardiologist": cardio,
            "neurologist": neuro,    "dermatologist": derma,
            "plain_summary": plain,
            "name": patient_name, "age": patient_age, "gender": patient_gender,
            "has_image": image_data is not None,
            "has_audio": audio_data is not None,
            "timestamp": datetime.now().strftime("%b %d, %Y  %H:%M"),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Analysis error: {exc}")
        raise HTTPException(500, str(exc))
    finally:
        for p in tmp_files:
            try: os.unlink(p)
            except: pass


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not READY:
        raise HTTPException(503, "AI not initialised")
    if len(req.question.strip()) < 2:
        raise HTTPException(400, "Question too short")
    try:
        parts = []
        if req.analysis_context and len(req.analysis_context.strip()) > 50:
            parts.append(
                f"Specialist analysis already performed:\n\n{req.analysis_context}\n\n---\n"
                "Answer the follow-up question specifically based on this analysis."
            )
        else:
            parts.append(
                "You are a healthcare AI assistant. No prior analysis — "
                "answer general health questions and recommend professional consultation."
            )
        if req.history:
            lines = []
            for item in req.history[-3:]:
                if isinstance(item, (list, tuple)) and len(item) == 2 and item[0] and item[1]:
                    lines.append(f"Patient: {item[0]}\nAssistant: {item[1]}")
            if lines:
                parts.append("Recent conversation:\n" + "\n".join(lines))
        parts.append(f"Patient ({req.patient_name}) asks: {req.question}")
        response = healthcare_ai.get_health_recommendation("\n\n".join(parts))
        return {"success": True, "response": response}
    except Exception as exc:
        logger.error(f"Chat error: {exc}")
        raise HTTPException(500, str(exc))


# Serve React — must be last
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0",
                port=int(os.getenv("PORT", 7860)), log_level="info")
