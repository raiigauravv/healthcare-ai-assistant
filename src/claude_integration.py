"""
Healthcare AI Assistant - Claude Integration
Uses Claude (claude-3-5-haiku) for patient dialogue / follow-up chat.
Falls back gracefully when ANTHROPIC_API_KEY is not set.
"""

import os
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "You are an empathetic healthcare AI assistant for educational purposes only. "
    "You do not provide medical diagnoses or prescriptions. "
    "Always recommend professional medical consultation. "
    "Be warm, clear, and reassuring while being medically responsible."
)

_CHAT_MODEL = "claude-3-5-haiku-20241022"   # fast, cost-effective for dialogue


class ClaudePatientAssistant:
    """Claude-powered conversational assistant for patient dialogue."""

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set — Claude patient dialogue disabled.")
            self.available = False
            self.client = None
        else:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.available = True
                logger.info(f"Claude client initialised ({_CHAT_MODEL})")
            except ImportError:
                logger.error("anthropic package not installed.")
                self.available = False
                self.client = None
            except Exception as exc:
                logger.error(f"Claude init error: {exc}")
                self.available = False
                self.client = None

    def chat(
        self,
        question: str,
        patient_name: str = "Patient",
        analysis_context: str = "",
        history: List[Tuple[str, str]] | None = None,
    ) -> str:
        """
        Send a message to Claude and return the reply.

        Parameters
        ----------
        question          : The patient's current question.
        patient_name      : Patient's first name for personalised responses.
        analysis_context  : Plain-text summary from the specialist analysis (if any).
        history           : Last few (user, assistant) pairs from the conversation.
        """
        if not self.available:
            return None   # Caller should fall back to OpenAI

        messages = []

        # Inject history (last 3 turns)
        for user_msg, asst_msg in (history or [])[-3:]:
            if user_msg and asst_msg:
                messages.append({"role": "user",    "content": str(user_msg)})
                messages.append({"role": "assistant","content": str(asst_msg)})

        # Current turn
        messages.append({"role": "user", "content": question})

        # Build system prompt — include analysis context when present
        if analysis_context and len(analysis_context.strip()) > 50:
            system = (
                f"{DISCLAIMER}\n\n"
                "A multi-specialist AI analysis has already been performed for this patient. "
                "The full analysis is shown below. Answer the patient's follow-up question "
                "specifically and accurately, drawing on the analysis results:\n\n"
                f"{analysis_context}"
            )
        else:
            system = (
                f"{DISCLAIMER}\n\n"
                "No prior analysis has been run. Answer general health questions helpfully, "
                "educationally, and always recommend professional consultation for specific concerns."
            )

        try:
            resp = self.client.messages.create(
                model=_CHAT_MODEL,
                max_tokens=600,
                system=system,
                messages=messages,
            )
            return resp.content[0].text.strip()
        except Exception as exc:
            logger.error(f"Claude chat error: {exc}")
            return None   # Caller falls back to OpenAI
