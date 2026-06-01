---
title: Healthcare AI Assistant
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# 🏥 Healthcare AI Assistant

A multimodal AI-powered healthcare assistant that delivers instant multi-specialist analysis from text descriptions, medical images, and voice recordings. Built with a modern React frontend and FastAPI backend, deployed live on Hugging Face Spaces.

![Status](https://img.shields.io/badge/Status-Live%20Demo-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![React](https://img.shields.io/badge/Frontend-React%2018-61dafb)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o%20Vision%20%2B%20Whisper-purple)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude%203.5%20Haiku-orange)

## 🚀 Live Demo

**[Try it on Hugging Face Spaces →](https://huggingface.co/spaces/gauravvraii/healthcare-ai-assistant)**

---

## ⚠️ Medical Disclaimer

This is a **demonstration tool for educational purposes only**. It does not provide medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional. In emergencies, contact emergency services immediately.

---

## ✨ Features

### 🤖 Four AI Specialist Agents
Each analysis runs simultaneously through four GPT-4o powered specialists:
| Specialist | Focus |
|---|---|
| 🩺 General Physician | Overall assessment, differential diagnosis, self-care |
| ❤️ Cardiologist | Cardiovascular risk, heart-related symptoms, ECG flags |
| 🧠 Neurologist | Neurological assessment, headache, nervous system red flags |
| 🔬 Dermatologist | Skin condition analysis, visual lesion assessment |

### 🖼️ GPT-4o Vision — Medical Image Analysis
- Upload any medical photo (skin conditions, rashes, nail changes, wounds)
- All four specialists examine the image via GPT-4o Vision
- Magic-byte format detection (JPEG / PNG / WEBP / GIF) — no extension guessing
- Automatic conversion for non-standard formats

### 🎤 Whisper ASR — Voice Input
- Record or upload audio (MP3, WAV, M4A, OGG, FLAC, WEBM)
- Transcribed via OpenAI Whisper-1
- Transcription merged with typed symptoms for richer analysis

### 💬 Claude — Patient Dialogue
- Follow-up chat powered by **Claude 3.5 Haiku** (Anthropic)
- Fully context-aware: remembers the specialist analysis and conversation history
- Falls back to GPT-4o-mini if `ANTHROPIC_API_KEY` is not set
- Chat messages show which model responded (Claude 🟠 vs GPT-4o-mini 🔵)

### 🎨 Modern React UI
- Animated hero with live status indicator
- Tabbed results panel — one tab per specialist + summary
- Confidence score bars, shimmer skeletons, WhatsApp-style chat bubbles
- Fully responsive (mobile-friendly)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│            React 18 SPA (frontend/)          │
│  Tailwind CSS · Babel Standalone · CDN only  │
└───────────────────┬─────────────────────────┘
                    │ fetch /api/*
┌───────────────────▼─────────────────────────┐
│           FastAPI Backend (main.py)          │
│                                             │
│  POST /api/analyze ──► AgentCoordinator     │
│  POST /api/chat    ──► ClaudePatientAssist. │
│  GET  /api/health                           │
│  POST /api/test-vision  (diagnostic)        │
└───────┬──────────────────────┬──────────────┘
        │                      │
┌───────▼──────┐    ┌──────────▼──────────────┐
│  OpenAI API  │    │     Anthropic API        │
│              │    │                         │
│ GPT-4o       │    │ claude-3-5-haiku         │
│  └ Vision ×4 │    │  └ patient dialogue      │
│ Whisper-1    │    └─────────────────────────┘
│ GPT-4o-mini  │
│  └ fallback  │
└──────────────┘
```

### File Structure
```
healthcare-ai-assistant/
├── main.py                      # FastAPI app — all API routes
├── Dockerfile                   # Docker build (python:3.10-slim + ffmpeg)
├── requirements.txt             # Python dependencies
├── frontend/
│   └── index.html               # Complete React SPA (single file)
└── src/
    ├── agents.py                # AgentCoordinator + 4 SPECIALISTS dict
    ├── claude_integration.py    # ClaudePatientAssistant (chat)
    ├── openai_integration.py    # OpenAIHealthcareAssistant (analysis + Whisper)
    ├── ingestion.py             # Image / audio / text ingestion utilities
    └── preprocess.py            # Text preprocessing
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.10+
- `ffmpeg` installed (`brew install ffmpeg` on macOS)
- OpenAI API key (required)
- Anthropic API key (optional — enables Claude chat)

### Install & Run
```bash
git clone https://github.com/raiigauravv/healthcare-ai-assistant
cd healthcare-ai-assistant

pip install -r requirements.txt

# Create .env with your keys
echo "OPENAI_API_KEY=sk-..." > .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env   # optional

python main.py
# Open http://localhost:7860
```

### Docker
```bash
docker build -t healthcare-ai .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=sk-... \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  healthcare-ai
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | GPT-4o Vision + Whisper + GPT-4o-mini |
| `ANTHROPIC_API_KEY` | Optional | Claude 3.5 Haiku for patient chat |

On Hugging Face Spaces: **Settings → Variables and secrets → New secret**

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 (CDN), Tailwind CSS (CDN), Babel Standalone |
| Backend | FastAPI, Uvicorn, Python 3.10 |
| Analysis AI | OpenAI GPT-4o (Vision) |
| Audio AI | OpenAI Whisper-1 |
| Chat AI | Anthropic Claude 3.5 Haiku |
| Image handling | Pillow, magic-byte format detection |
| Audio handling | librosa, soundfile |
| Deployment | Docker on Hugging Face Spaces |

---

## 🔌 API Reference

### `GET /api/health`
Returns system status.
```json
{ "ok": true, "ai_ready": true, "api_configured": true, "claude_available": true }
```

### `POST /api/analyze`
Multipart form — runs all 4 specialist agents.

| Field | Type | Required |
|---|---|---|
| `symptoms` | string (≥10 chars) | ✅ |
| `patient_name` | string | ✅ |
| `patient_age` | int (default 30) | — |
| `patient_gender` | string | — |
| `image` | file (JPEG/PNG/WEBP/GIF) | — |
| `audio` | file (MP3/WAV/M4A/…) | — |

### `POST /api/chat`
JSON body — Claude-powered follow-up chat.
```json
{
  "question": "What precautions should I take?",
  "patient_name": "John",
  "analysis_context": "<plain text from previous analysis>",
  "history": [["prev question", "prev answer"]]
}
```

### `POST /api/test-vision` *(diagnostic)*
Upload a single image, get raw GPT-4o Vision response. Useful for verifying Vision API access.

---

## 🛡️ Privacy & Security

- **No persistent storage** — patient data is processed in memory and discarded after each request
- **No logging of patient content** — only operational logs (image size, model used)
- **API keys via environment secrets** — never hardcoded
- **HTTPS only** on Hugging Face Spaces deployment

---

## 📈 Roadmap

- [ ] Additional specialists (Orthopaedics, Gastroenterology, Psychiatry)
- [ ] PDF report export
- [ ] Multi-language support
- [ ] Session history (opt-in, client-side only)
- [ ] Streaming responses for faster perceived latency

---

## 👤 Author

**Gaurav** · [HuggingFace](https://huggingface.co/gauravvraii) · [GitHub](https://github.com/raiigauravv)

---

*Built with GPT-4o Vision, Whisper ASR, Claude 3.5 Haiku, React, and FastAPI — deployed on Hugging Face Spaces.*
