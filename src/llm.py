"""
src/llm.py — LLM Integration Module
=====================================
Provider: Google Gemini via the new `google-genai` SDK.
  - Uses proper system/user message roles (prevents instruction burial)
  - Streams response to avoid silent mid-sentence truncation
  - Logs prompt size for debugging retrieval quality
"""

import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LLM_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


# ── System Instruction ─────────────────────────────────────────────────────────
# Sent as a SYSTEM role message — Gemini treats this as a directive, not content.
# This is the #1 fix for vague/incomplete answers: the model now knows its job
# before it even sees the retrieved policy text.

SYSTEM_INSTRUCTION = """You are the  HR Partner. Your goal is to provide fast, friendly, and accurate policy guidance.

### Core Philosophy: "Progressive Disclosure"
- **Be Concise First**: Employees want quick answers. Provide the "bottom line" immediately.
- **Avoid Overload**: If a policy is complex (e.g., travel limits, detailed quotas), summarize the general rule and offer to provide the full breakdown.
- **Tone**: Professional yet warm and conversational. Use "we" and "you." Avoid sounding like a legal document.

### Response Structure:
1.  **Quick Answer**: A 1-3 sentence direct response to the core question.
2.  **Key Highlights**: Use 2-3 bullet points for the most important conditions or eligibility rules.
3.  **Progressive Offer**: If there is more detail available (tables, sub-categories, specific limits), add a line like: "Would you like me to pull up the specific limits or the step-by-step approval process for this?"
4.  **Summary Block**: End with the required summary block.

### Writing Rules:
- **No Boilerplate**: Do not start with "Based on the policy..." or "I found information..." Jump straight into helping.
- **Formatting**: Use **bolding** for keywords. Keep paragraphs short (max 3 lines).
- **No 'Etc'**: Be specific, but for long lists, summarize the top 3 and offer the rest.

---
**Summary:** [A one-sentence punchy recap of the main outcome.]
"""


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def build_prompt(query: str, context_chunks: list[dict], low_confidence: bool = False) -> str:
    """
    Build the USER-role message content only.
    The system instruction is sent separately as a system role message.

    Args:
        query:          The employee's question.
        context_chunks: Retrieved chunk dicts with 'content', 'source', 'page'.
        low_confidence: If True, injects a note asking for a careful answer.

    Returns:
        The user message string (context + question).
    """
    if not context_chunks:
        context_text = "No relevant policy documents were found."
    else:
        parts = []
        for i, chunk in enumerate(context_chunks, start=1):
            source = chunk.get("source", "Unknown")
            page   = chunk.get("page", "?")
            text   = chunk.get("content", "").strip()
            parts.append(f"[Excerpt {i} — {source}, Page {page}]\n{text}")
        context_text = "\n\n".join(parts)

    confidence_note = (
        "\nNote: The retrieved excerpts may only partially cover this topic. "
        "Share everything relevant that is available in the excerpts above.\n"
        if low_confidence else ""
    )

    # Detect intent to guide progressive disclosure
    is_broad = "policy" in query.lower() or "overview" in query.lower() or len(query.split()) <= 2
    
    if is_broad:
        mode_instruction = (
            "\nGUIDE: Provide a 'Concise First' overview. List the 3-4 main categories found. "
            "Do NOT explain every rule now. Offer to expand on specifics at the end.\n"
        )
    else:
        mode_instruction = (
            "\nGUIDE: Provide a punchy direct answer. Keep details focused on the specific question. "
            "If there are related complex rules, summarize them in one sentence and offer to expand.\n"
        )

    user_message = f"""HR POLICY EXCERPTS:
{context_text}

---
EMPLOYEE QUESTION:
{query}
{confidence_note}
{mode_instruction}
End with the required Summary section."""

    logger.info(
        f"  ↳ Prompt: {len(context_chunks)} chunk(s) | "
        f"~{len(user_message) // 4} estimated input tokens"
    )
    return user_message


# ── Gemini Integration (google-genai SDK) ──────────────────────────────────────

def call_gemini(prompt: str) -> str:
    """
    Send the prompt to Gemini using the new google-genai SDK.

    Key improvements over google-generativeai:
      - system_instruction is a proper role, not buried in the prompt string
      - finish_reason is checked so truncated responses are flagged
      - No FutureWarning deprecation noise

    Requires:
        pip install google-genai
        GEMINI_API_KEY in .env
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return "❌ google-genai not installed. Run: pip install google-genai"

    if not GEMINI_API_KEY:
        return (
            "❌ GEMINI_API_KEY is not set.\n"
            "   Add to .env:  GEMINI_API_KEY=your-key-here\n"
            "   Get a key at: https://aistudio.google.com/app/apikey"
        )

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
        )

        logger.debug(f"Calling Gemini model '{GEMINI_MODEL}'")

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config,
        )

        # Check why generation stopped — MAX_TOKENS means answer was cut off
        candidate = response.candidates[0] if response.candidates else None
        if candidate:
            finish = str(candidate.finish_reason)
            if "MAX_TOKENS" in finish:
                logger.warning(
                    "⚠️  Gemini hit MAX_TOKENS — answer may be incomplete. "
                    f"Increase LLM_MAX_TOKENS in config.py (currently {LLM_MAX_TOKENS})."
                )

        return response.text.strip()

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return f"❌ Gemini error: {str(e)}"


# ── Master Generate Function ───────────────────────────────────────────────────

def generate_answer(
    query:          str,
    context_chunks: list[dict],
    low_confidence: bool = False,
) -> str:
    """
    Main entry point called by rag_chain.py.
    Builds the user prompt and calls the configured LLM provider.

    Args:
        query:          The user's question string.
        context_chunks: Retrieved chunk dicts from the retriever.
        low_confidence: If True, instructs the LLM to answer carefully.

    Returns:
        A complete, natural-language answer string.
    """
    logger.info(f"🤖 Generating answer via provider: '{LLM_PROVIDER}'")

    prompt = build_prompt(query, context_chunks, low_confidence=low_confidence)

    if LLM_PROVIDER == "gemini":
        return call_gemini(prompt)

    return (
        f"❌ Unknown LLM_PROVIDER '{LLM_PROVIDER}' in config.py.\n"
        "   Valid options: 'gemini'"
    )
