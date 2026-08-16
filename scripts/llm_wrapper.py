# LLM wrapper for answering questions with retrieved context.
# Supports both legal documents and navigation/procedural guidance.

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load .env from backend first, then fallback to project root
_ROOT = Path(__file__).resolve().parents[1]
_ENV_PATH = _ROOT / "backend" / ".env"
if not _ENV_PATH.exists():
    _ENV_PATH = _ROOT / ".env"
load_dotenv(_ENV_PATH)

DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


def detect_query_type(question: str, context_chunks: List[str] = None) -> str:
    """
    Detect whether the query is about legal information or procedural navigation.
    Returns: 'legal', 'navigation', or 'mixed'
    """
    question_lower = question.lower()
    
    # Navigation indicators
    nav_keywords = [
        "how to", "how do i", "how can i", "steps to", "process for",
        "where to", "which office", "documents required", "documents needed",
        "apply for", "get a", "obtain", "register for", "registration",
        "renew", "renewal", "what documents", "how much", "cost of",
        "time required", "how long", "procedure", "kaha", "kasari",
        "office address", "phone number", "contact"
    ]
    
    # Legal indicators  
    legal_keywords = [
        "what is", "what are", "define", "definition", "according to",
        "section", "article", "act says", "law says", "provision",
        "constitutional", "legal rights", "punishment", "penalty",
        "offense", "crime", "rights", "duties", "fundamental"
    ]
    
    nav_score = sum(1 for kw in nav_keywords if kw in question_lower)
    legal_score = sum(1 for kw in legal_keywords if kw in question_lower)
    
    if nav_score > legal_score:
        return "navigation"
    elif legal_score > nav_score:
        return "legal"
    else:
        return "mixed"


def generate_answer(
    question: str,
    context_chunks: List[str],
    chunk_metadata: List[Dict] = None,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:

    # HuggingFace Inference API call 

    api_key = HUGGINGFACE_API_KEY
    if not api_key:
        raise RuntimeError("api key not found")

    model = model or DEFAULT_LLM_MODEL
    client = InferenceClient(token=api_key)

    # Detect query type for better prompting
    query_type = detect_query_type(question, context_chunks)
    
    # Build context with source attribution
    if chunk_metadata:
        context_parts = []
        for i, (chunk, meta) in enumerate(zip(context_chunks, chunk_metadata)):
            source = meta.get("title") or meta.get("filename") or "Unknown Source"
            context_parts.append(f"[Source: {source}]\n{chunk}")
        context = "\n\n---\n\n".join(context_parts)
    else:
        context = "\n\n---\n\n".join(context_chunks)
    
    # Dynamic system prompt based on query type
    if query_type == "navigation":
        system_prompt = """You are a helpful government services assistant for Nepal.
Your task is to guide users through government procedures, services, and processes.

**CRITICAL RULES:**
1. ONLY use information from the provided context. Do NOT use general knowledge.
2. Your response MUST end with the TL;DR summary. Do NOT write anything after TL;DR.
3. NEVER add a "Contact Information" section.
4. NEVER say "unfortunately" or mention missing information.

Guidelines:
1. Provide clear, step-by-step instructions when explaining procedures.
2. Include relevant details from the context naturally in your answer.
3. Be practical and actionable.
4. Use numbered lists for steps and bullet points for documents.
5. End with "**TL;DR:**" summary - this MUST be your final output."""

    elif query_type == "legal":
        system_prompt = """You are a legal assistant specializing in Nepali law.
Your task is to answer questions based STRICTLY on the provided legal document excerpts.

**CRITICAL RULES:**
1. ONLY use information from the provided context. Do NOT use general knowledge.
2. Your response MUST end with the TL;DR summary. Do NOT write anything after TL;DR.
3. NEVER add a "Contact Information" section.
4. NEVER say "unfortunately" or mention missing information.
5. NEVER add suggestions to "contact authorities" or "consult professionals".

Guidelines:
1. Answer using ONLY information from the provided context.
2. Cite the source document name with section/article (e.g., "According to Section 38 of the Constitution of Nepal...").
3. Be thorough but concise.
4. End with "**TL;DR:**" summary - this MUST be your final output."""

    else:  # mixed
        system_prompt = """You are a helpful assistant specializing in Nepali law and government services.

**CRITICAL RULES:**
1. ONLY use information from the provided context. Do NOT use general knowledge.
2. Your response MUST end with the TL;DR summary. Do NOT write anything after TL;DR.
3. NEVER add a "Contact Information" section.
4. NEVER say "unfortunately" or mention missing information.
5. NEVER add suggestions to "contact authorities" or "consult professionals".

Guidelines:
1. For legal questions: Cite the source document name with section/article.
2. For procedural questions: Provide step-by-step guidance when available.
3. Include relevant details from the context naturally.
4. End with "**TL;DR:**" summary - this MUST be your final output."""

    user_message = f"""Context:
{context}

---

Question: {question}

Answer using ONLY the context above. Cite sources. End with **TL;DR:** - nothing after it."""

    # Use chat_completion for conversational models like Llama
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return response.choices[0].message.content.strip()


__all__ = ["generate_answer", "DEFAULT_LLM_MODEL", "detect_query_type"]
