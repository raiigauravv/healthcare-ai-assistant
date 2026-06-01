"""
Healthcare AI Assistant — FastAPI backend
Serves React SPA on / and AI endpoints on /api/*
"""
import base64
import io
import os
import tempfile
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from src.ingestion import ingest_text, ingest_image, ingest_audio
from src.preprocess import preprocess_text
from src.openai_integration import OpenAIHealthcareAssistant
from src.claude_integration import ClaudePatientAssistant
from src.agents import AgentCoordinator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"===== Startup {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")

try:
    healthcare_ai = OpenAIHealthcareAssistant()
    claude_assistant = ClaudePatientAssistant()
    agent_coordinator = AgentCoordinator()
    READY = True
    logger.info(
        f"AI components initialised — "
        f"Claude dialogue: {'ON' if claude_assistant.available else 'OFF (no ANTHROPIC_API_KEY)'}, "
        f"OpenAI: {'mock' if healthcare_ai.mock_mode else 'ON'}"
    )
except Exception as exc:
    logger.error(f"AI init failed: {exc}")
    READY = False
    claude_assistant = None

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

@app.get("/api/debug-vision")
async def debug_vision():
    """
    No upload needed. Generates a test image in memory, runs it through the
    EXACT same _call_specialist code path used during analysis, and returns
    the raw response. Tells us definitively if the agents pipeline works.
    """
    if not READY or healthcare_ai.mock_mode:
        raise HTTPException(503, "OpenAI API key not configured")
    try:
        # Build a small but non-trivial test image
        img = Image.new("RGB", (200, 200), color=(220, 100, 80))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([40, 40, 160, 160], fill=(80, 180, 80))
        draw.ellipse([70, 70, 130, 130], fill=(60, 100, 220))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        raw = buf.getvalue()
        img_b64 = base64.b64encode(raw).decode("utf-8")
        img_mime = "image/jpeg"

        logger.info(f"debug-vision: generated {len(raw)} byte JPEG, b64={len(img_b64)} chars")

        # Call _call_specialist directly — same path as /api/analyze
        result = agent_coordinator._call_specialist(
            "General Physician",
            "an experienced family medicine physician",
            "Patient: Test, 30 year old Male.\nSymptoms: test symptoms",
            img_b64,
            img_mime,
        )

        return {
            "ok": True,
            "image_bytes": len(raw),
            "b64_chars": len(img_b64),
            "mime": img_mime,
            "data_url_prefix": f"data:{img_mime};base64,{img_b64[:20]}...",
            "specialist_saw_image": "unable to view" not in result.get("analysis","").lower(),
            "analysis_snippet": result.get("analysis","")[:200],
            "client_initialized": agent_coordinator._client is not None,
            "model": agent_coordinator._MODEL,
        }
    except Exception as exc:
        logger.error(f"debug-vision error: {exc}", exc_info=True)
        return {"error": str(exc)}


@app.post("/api/test-vision")
async def test_vision(image: UploadFile = File(...)):
    """
    Diagnostic endpoint — upload any image and get a raw GPT-4o Vision response.
    Bypasses all preprocessing so we can isolate exactly where Vision breaks.
    """
    if not READY or healthcare_ai.mock_mode:
        raise HTTPException(503, "OpenAI API key not configured")
    try:
        raw = await image.read()
        if not raw:
            return {"error": "0 bytes received — upload failed"}

        ext  = os.path.splitext(image.filename)[1].lower().lstrip(".")
        mime = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png",
                "webp":"image/webp","gif":"image/gif"}.get(ext, "image/jpeg")
        img_b64 = base64.b64encode(raw).decode("utf-8")

        logger.info(f"test-vision: {len(raw)//1024}KB raw  {len(img_b64)//1024}KB b64  mime={mime}")

        resp = healthcare_ai.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe exactly what you see in this image in 2-3 sentences."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime};base64,{img_b64}",
                        "detail": "low",
                    }},
                ],
            }],
            max_tokens=200,
        )
        return {
            "ok": True,
            "original_kb": len(raw) // 1024,
            "b64_kb": len(img_b64) // 1024,
            "mime": mime,
            "gpt4o_says": resp.choices[0].message.content,
        }
    except Exception as exc:
        logger.error(f"test-vision error: {exc}", exc_info=True)
        return {"error": str(exc)}


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "ai_ready": READY,
        "api_configured": READY and not healthcare_ai.mock_mode,
        "claude_available": READY and claude_assistant is not None and claude_assistant.available,
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

    image_b64: str | None = None   # base64-encoded image bytes — passed directly to agents
    image_mime: str = "image/jpeg" # MIME type for data URI
    image_data = None              # truthy flag — set when image loads OK
    audio_data = None
    tmp_files: list[str] = []

    try:
        # ── Image: detect format from magic bytes, encode raw bytes → base64 ───
        if image and image.filename:
            try:
                raw = await image.read()
                if not raw:
                    logger.warning(f"Image upload '{image.filename}' produced 0 bytes — skipping.")
                else:
                    # Detect format from magic bytes (not unreliable extension)
                    if raw[:3] == b'\xff\xd8\xff':
                        mime = "image/jpeg"
                    elif raw[:8] == b'\x89PNG\r\n\x1a\n':
                        mime = "image/png"
                    elif raw[:6] in (b'GIF87a', b'GIF89a'):
                        mime = "image/gif"
                    elif raw[:4] == b'RIFF' and raw[8:12] == b'WEBP':
                        mime = "image/webp"
                    else:
                        # Unsupported format (e.g. HEIC) — convert via PIL to JPEG
                        logger.info(f"Unknown magic bytes, converting via PIL: {raw[:8].hex()}")
                        pil = Image.open(io.BytesIO(raw)).convert("RGB")
                        buf = io.BytesIO()
                        pil.save(buf, format="JPEG", quality=90)
                        raw = buf.getvalue()
                        mime = "image/jpeg"

                    image_b64 = base64.b64encode(raw).decode("utf-8")
                    image_mime = mime
                    image_data = True   # flag only
                    logger.info(
                        f"Image ready: {len(raw)//1024}KB  mime={mime}  "
                        f"b64_len={len(image_b64)}  file='{image.filename}'"
                    )
            except Exception as exc:
                logger.error(f"Image read error ('{image.filename}'): {exc}", exc_info=True)
                image_b64 = None
                image_data = None

        # ── Audio: write to temp file, flush, then read (avoids partial-read) ─
        if audio and audio.filename:
            suffix = os.path.splitext(audio.filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(await audio.read())
                f.flush()          # push Python buffer → OS before closing
                tmp_files.append(f.name)
            # File is closed & fully on disk now — safe to read
            audio_data = ingest_audio(tmp_files[-1])
            logger.info(f"Audio ingested: {tmp_files[-1]}, data={audio_data is not None}")

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
            "has_image": image_b64 is not None,
            "has_audio": audio_data is not None,
        }

        logger.info(
            f"Calling agents — image_b64={'YES ('+str(len(image_b64)//1024)+'KB)' if image_b64 else 'NO'}, "
            f"audio={audio_data is not None}"
        )
        # image_b64 + image_mime passed directly to each specialist's GPT-4o Vision call
        results = agent_coordinator.analyze_with_agents(ctx, full_symptoms, image_b64, audio_data, image_mime)
        gp     = results.get("General Physician", {})
        cardio = results.get("Cardiologist", {})
        neuro  = results.get("Neurologist", {})
        derma  = results.get("Dermatologist", {})

        plain = (
            f"Patient: {patient_name}, {patient_age}yo {patient_gender}\n"
            f"Symptoms: {full_symptoms}\n"
            + (f"Image: uploaded and examined by GPT-4o Vision\n" if image_b64 else "")
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
            "has_image": image_b64 is not None,
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
        # Normalise history into list-of-tuples for Claude
        history_pairs = []
        if req.history:
            for item in req.history[-3:]:
                if isinstance(item, (list, tuple)) and len(item) == 2 and item[0] and item[1]:
                    history_pairs.append((str(item[0]), str(item[1])))

        # ── 1. Try Claude (patient dialogue model) ──────────────────────────
        response = None
        model_used = "openai"
        if claude_assistant and claude_assistant.available:
            response = claude_assistant.chat(
                question=req.question,
                patient_name=req.patient_name,
                analysis_context=req.analysis_context,
                history=history_pairs,
            )
            if response:
                model_used = "claude"
                logger.info("Chat handled by Claude")

        # ── 2. Fall back to OpenAI gpt-4o-mini ──────────────────────────────
        if not response:
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
            if history_pairs:
                lines = [f"Patient: {u}\nAssistant: {a}" for u, a in history_pairs]
                parts.append("Recent conversation:\n" + "\n".join(lines))
            parts.append(f"Patient ({req.patient_name}) asks: {req.question}")
            response = healthcare_ai.get_health_recommendation("\n\n".join(parts))
            logger.info("Chat handled by OpenAI (fallback)")

        return {"success": True, "response": response, "model": model_used}
    except Exception as exc:
        logger.error(f"Chat error: {exc}")
        raise HTTPException(500, str(exc))


# Serve React — must be last
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0",
                port=int(os.getenv("PORT", 7860)), log_level="info")
