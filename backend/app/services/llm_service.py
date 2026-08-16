"""
LLM service for generating answers using HuggingFace Inference API.
Matches scripts/llm_wrapper.py behavior with query type detection and context-aware prompts.
"""

import logging
import os
from typing import List, Optional, Dict
from huggingface_hub import InferenceClient

from app.core.config import settings

logger = logging.getLogger(__name__)

# Query type keywords matching scripts/llm_wrapper.py
NAVIGATION_KEYWORDS = [
    "how to", "how do i", "how can i", "steps to", "process for",
    "where to", "which office", "documents required", "documents needed",
    "apply for", "get a", "obtain", "register for", "registration",
    "renew", "renewal", "what documents", "how much", "cost of",
    "time required", "how long", "procedure", "kaha", "kasari",
    "office address", "phone number", "contact"
]

LEGAL_KEYWORDS = [
    "what is", "what are", "define", "definition", "according to",
    "section", "article", "act says", "law says", "provision",
    "constitutional", "legal rights", "punishment", "penalty",
    "offense", "crime", "rights", "duties", "fundamental"
]

# System prompts for different query types (matching scripts/llm_wrapper.py)
NAVIGATION_PROMPT = """You are a helpful government services assistant for Nepal.
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

LEGAL_PROMPT = """You are a legal assistant specializing in Nepali law.
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

MIXED_PROMPT = """You are a helpful assistant specializing in Nepali law and government services.

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


class LLMService:
    """
    Handles LLM inference using HuggingFace Inference API.
    Implements singleton pattern and query type detection.
    Matches scripts/llm_wrapper.py behavior.
    """
    
    _instance = None
    _client = None
    
    def __new__(cls):
        """Singleton pattern to ensure client is initialized only once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the HuggingFace Inference API client."""
        if LLMService._client is None:
            self._load_client()
    
    def _load_client(self):
        """Initialize HuggingFace Inference API client."""
        try:
            api_key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
            
            if not api_key:
                raise RuntimeError(
                    "HuggingFace API key not found. "
                    "Set HF_TOKEN or HUGGINGFACE_API_KEY environment variable."
                )
            
            LLMService._client = InferenceClient(api_key=api_key)
            logger.info(f"Initialized HuggingFace Inference API client")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise RuntimeError(f"Failed to initialize LLM client: {e}")
    
    @property
    def client(self) -> InferenceClient:
        """Get the API client."""
        if LLMService._client is None:
            self._load_client()
        return LLMService._client
    
    @staticmethod
    def detect_query_type(question: str) -> str:
        """
        Detect query type: 'legal', 'navigation', or 'mixed'.
        Matches scripts/llm_wrapper.py detect_query_type() logic.
        
        Args:
            question: User's question
            
        Returns:
            Query type: 'legal', 'navigation', or 'mixed'
        """
        question_lower = question.lower()
        
        # Count keyword matches
        nav_score = sum(1 for kw in NAVIGATION_KEYWORDS if kw in question_lower)
        legal_score = sum(1 for kw in LEGAL_KEYWORDS if kw in question_lower)
        
        if nav_score > legal_score:
            return "navigation"
        elif legal_score > nav_score:
            return "legal"
        else:
            return "mixed"
    
    def get_system_prompt(self, query_type: str) -> str:
        """
        Get the appropriate system prompt based on query type.
        
        Args:
            query_type: One of 'legal', 'navigation', 'mixed'
            
        Returns:
            System prompt string
        """
        if query_type == "navigation":
            return NAVIGATION_PROMPT
        elif query_type == "legal":
            return LEGAL_PROMPT
        else:
            return MIXED_PROMPT
    
    def build_rag_prompt(
        self, 
        question: str, 
        context_chunks: List[str],
        chunk_metadata: Optional[List[Dict]] = None
    ) -> tuple[str, str]:
        """
        Build a RAG prompt with context and question.
        Matches scripts/llm_wrapper.py context formatting.
        
        Args:
            question: User's question
            context_chunks: List of relevant text chunks
            chunk_metadata: Optional metadata for each chunk (for source attribution)
            
        Returns:
            Tuple of (system_prompt, user_message)
        """
        # Detect query type and get appropriate system prompt
        query_type = self.detect_query_type(question)
        system_prompt = self.get_system_prompt(query_type)
        
        # Format context with source attribution (matching scripts/llm_wrapper.py)
        if chunk_metadata:
            context_parts = []
            for i, (chunk, meta) in enumerate(zip(context_chunks, chunk_metadata)):
                source = meta.get("title") or meta.get("filename") or "Unknown Source"
                context_parts.append(f"[Source: {source}]\n{chunk}")
            context = "\n\n---\n\n".join(context_parts)
        else:
            context = "\n\n---\n\n".join(context_chunks)
        
        # Build user message
        user_message = f"""Context from Nepali Legal Documents:

{context}

Based on the above context, please answer the following question. If the answer cannot be found in the context, say so clearly.

Question: {question}"""
        
        return system_prompt, user_message
    
    def generate_answer(
        self, 
        question: str,
        context_chunks: List[str],
        chunk_metadata: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate an answer using HuggingFace Inference API.
        Matches scripts/llm_wrapper.py generate_answer() logic.
        
        Args:
            question: User's question
            context_chunks: List of relevant context chunks
            chunk_metadata: Optional metadata for source attribution
            model: Override model name (default from settings)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Generated answer string
        """
        try:
            model = model or settings.LLM_MODEL
            
            # Build RAG prompt with appropriate system prompt
            system_prompt, user_message = self.build_rag_prompt(
                question, 
                context_chunks,
                chunk_metadata
            )
            
            logger.info(f"Generating answer using {model} (query_type: {self.detect_query_type(question)})")
            
            # Call HuggingFace Inference API
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info("Answer generated successfully")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I apologize, but I encountered an error generating the response: {str(e)}"
    
    def is_loaded(self) -> bool:
        """Check if the LLM client is properly initialized."""
        return LLMService._client is not None
