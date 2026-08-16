"""
Text chunking utility for splitting documents into smaller pieces.
Implements various chunking strategies for optimal RAG performance.
"""

import logging
import re
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


class TextChunker:
    """
    Splits text into overlapping chunks for embedding and retrieval.
    Implements section-aware chunking matching scripts/clean.py strategy.
    Preserves legal document structure (Part/Chapter/Section/Article).
    """
    
    # Word-fixing dictionary matching scripts/clean.py (70+ patterns)
    WORD_FIXES = {
        r'\bA ct\b': 'Act',
        r'\bof fence': 'offence',
        r'\bOf fence': 'Offence',
        r'\ba ct\b': 'act',
        r'\bth is\b': 'this',
        r'\bTh is\b': 'This',
        r'\bth at\b': 'that',
        r'\bTh at\b': 'That',
        r'\bth e\b': 'the',
        r'\bTh e\b': 'The',
        r'\bwh ich\b': 'which',
        r'\bWh ich\b': 'Which',
        r'\bsh all\b': 'shall',
        r'\bSh all\b': 'Shall',
        r'\bwh o\b': 'who',
        r'\bWh o\b': 'Who',
        r'\ba nd\b': 'and',
        r'\bA nd\b': 'And',
        r'\bor der\b': 'order',
        r'\bOr der\b': 'Order',
        r'\bin to\b': 'into',
        r'\bIn to\b': 'Into',
        r'\bhe lp': 'help',
        r'\bHe lp': 'Help',
        r'\bshe lter': 'shelter',
        r'\bShe lter': 'Shelter',
        r'\ba ny\b': 'any',
        r'\bA ny\b': 'Any',
        r'\bma y\b': 'may',
        r'\bMa y\b': 'May',
        r'\bbe ing\b': 'being',
        r'\bBe ing\b': 'Being',
        r'\ba lso\b': 'also',
        r'\bA lso\b': 'Also',
        r'\bin cludes?\b': 'include',
        r'\bIn cludes?\b': 'Include',
        r'\ba uthoriz': 'authoriz',
        r'\bA uthoriz': 'Authoriz',
        r'\ba uthority': 'authority',
        r'\bA uthority': 'Authority',
        r'\ba pprehension': 'apprehension',
        r'\bA pprehension': 'Apprehension',
        r'\ba ssist': 'assist',
        r'\bA ssist': 'Assist',
        r'\ba mend': 'amend',
        r'\bA mend': 'Amend',
        r'\ba mong': 'among',
        r'\bA mong': 'Among',
        r'\ba pplic': 'applic',
        r'\bA pplic': 'Applic',
        r'\ba rticle': 'article',
        r'\bA rticle': 'Article',
        r'\ba rms\b': 'arms',
        r'\bA rms\b': 'Arms',
        r'\ba mmunition': 'ammunition',
        r'\bA mmunition': 'Ammunition',
        r'\ba ircraft': 'aircraft',
        r'\bA ircraft': 'Aircraft',
        r'\ba bandon': 'abandon',
        r'\bA bandon': 'Abandon',
        r'\ba betment': 'abetment',
        r'\bA betment': 'Abetment',
        r'\ba gainst': 'against',
        r'\bA gainst': 'Against',
        r'\ba djudic': 'adjudic',
        r'\bA djudic': 'Adjudic',
        r'\ba dult': 'adult',
        r'\bA dult': 'Adult',
        r'\bfor gery': 'forgery',
        r'\bFor gery': 'Forgery',
        r'\bimprison ment': 'imprisonment',
        r'\bImprison ment': 'Imprisonment',
        r'\bhe inous': 'heinous',
        r'\bHe inous': 'Heinous',
        r'\bat tempt': 'attempt',
        r'\bAt tempt': 'Attempt',
        r'\bin cest': 'incest',
        r'\bIn cest': 'Incest',
        r'\bin terest': 'interest',
        r'\bIn terest': 'Interest',
        r'\bin tent': 'intent',
        r'\bIn tent': 'Intent',
        r'\bmainta in\b': 'maintain',
        r'\bMainta in\b': 'Maintain',
        r'\bsup ply': 'supply',
        r'\bSup ply': 'Supply',
        r'\bthe refore': 'therefore',
        r'\bThe refore': 'Therefore',
        # Common concatenations
        r'\bofthe\b': 'of the',
        r'\btothe\b': 'to the',
        r'\binthe\b': 'in the',
        r'\bforthe\b': 'for the',
        r'\bandthe\b': 'and the',
        r'\borthe\b': 'or the',
        r'\bbythe\b': 'by the',
        r'\bonthe\b': 'on the',
        r'\batthe\b': 'at the',
        r'\basthe\b': 'as the',
        r'\bifthe\b': 'if the',
    }
    
    def __init__(
        self, 
        chunk_size: int = 2000,  # Default from scripts/clean.py
        chunk_overlap: int = 400,  # Default from scripts/clean.py
        min_chunk_size: int = 100  # Default from scripts/clean.py
    ):
        """
        Initialize the chunker with specified parameters.
        Matches scripts/clean.py prepare_for_rag() defaults.
        
        Args:
            chunk_size: Target size of each chunk (2000 default)
            chunk_overlap: Fixed overlap in characters (400 default)
            min_chunk_size: Minimum chunk size to keep (100 default)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
    
    def chunk_text(
        self, 
        text: str, 
        document_name: str = "unknown"
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks using section-aware strategy.
        Matches scripts/clean.py prepare_for_rag() logic.
        
        Args:
            text: The text to chunk
            document_name: Name of the source document for metadata
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for chunking: {document_name}")
            return []
        
        # Clean and fix text
        text = self._clean_text(text)
        text = self._fix_words(text)
        
        # Extract metadata before chunking
        metadata = self._extract_metadata(text, document_name)
        
        # Split by major sections (Part/Chapter/Section/Article)
        major_sections = re.split(
            r"\n(?=(?:Part|Chapter|Section|Article)\s*[-:]?\s*\d+)", 
            text
        )
        
        chunks = []
        chunk_index = 0
        current_accumulated = ""
        
        for section in major_sections:
            section = section.strip()
            if not section:
                continue
            
            # If section is very small, accumulate with next
            if len(section) < self.min_chunk_size:
                current_accumulated += "\n" + section if current_accumulated else section
                continue
            
            # Add any accumulated small sections
            if current_accumulated:
                section = current_accumulated + "\n" + section
                current_accumulated = ""
            
            if len(section) <= self.chunk_size:
                # Section fits in one chunk
                chunk_text = section.strip()
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = TextChunk(
                        text=chunk_text,
                        index=chunk_index,
                        start_char=0,
                        end_char=len(chunk_text),
                        metadata={
                            **metadata,
                            "chunk_index": chunk_index,
                            "char_count": len(chunk_text)
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            else:
                # Section is large, split with word-based overlap
                words = section.split()
                current_chunk = []
                current_length = 0
                
                for word in words:
                    current_chunk.append(word)
                    current_length += len(word) + 1
                    
                    if current_length >= self.chunk_size:
                        chunk_text = " ".join(current_chunk).strip()
                        if len(chunk_text) >= self.min_chunk_size:
                            chunk = TextChunk(
                                text=chunk_text,
                                index=chunk_index,
                                start_char=0,
                                end_char=len(chunk_text),
                                metadata={
                                    **metadata,
                                    "chunk_index": chunk_index,
                                    "char_count": len(chunk_text)
                                }
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                        
                        # Keep last part for overlap
                        overlap_words = int(len(current_chunk) * (self.chunk_overlap / self.chunk_size))
                        current_chunk = current_chunk[-overlap_words:]
                        current_length = sum(len(w) + 1 for w in current_chunk)
                
                if current_chunk:
                    remaining = " ".join(current_chunk).strip()
                    if len(remaining) >= self.min_chunk_size:
                        chunk = TextChunk(
                            text=remaining,
                            index=chunk_index,
                            start_char=0,
                            end_char=len(remaining),
                            metadata={
                                **metadata,
                                "chunk_index": chunk_index,
                                "char_count": len(remaining)
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1
        
        # Handle remaining accumulated content
        if current_accumulated and len(current_accumulated) >= self.min_chunk_size:
            if chunks:
                # Append to last chunk
                last_chunk = chunks[-1]
                last_chunk.text = last_chunk.text + "\n" + current_accumulated
                last_chunk.metadata["char_count"] = len(last_chunk.text)
            else:
                chunk = TextChunk(
                    text=current_accumulated,
                    index=chunk_index,
                    start_char=0,
                    end_char=len(current_accumulated),
                    metadata={
                        **metadata,
                        "chunk_index": 0,
                        "char_count": len(current_accumulated)
                    }
                )
                chunks.append(chunk)
        
        logger.info(
            f"Created {len(chunks)} chunks from {document_name} "
            f"(avg size: {sum(len(c.text) for c in chunks) // max(len(chunks), 1)} chars)"
        )
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text for better chunking.
        Matches scripts/clean.py logic.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove excessive whitespace while preserving paragraph breaks
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines]
        text = '\n'.join(cleaned_lines)
        
        return text.strip()
    
    def _fix_words(self, text: str) -> str:
        """
        Apply word-fix patterns to correct OCR artifacts.
        Matches scripts/clean.py WORD_FIXES logic.
        
        Args:
            text: Text to fix
            
        Returns:
            Text with fixes applied
        """
        for pattern, replacement in self.WORD_FIXES.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _extract_metadata(self, text: str, filename: str) -> Dict[str, str]:
        """
        Extract metadata from document.
        Matches scripts/clean.py extract_metadata() logic.
        
        Args:
            text: Document text
            filename: Source filename
            
        Returns:
            Metadata dictionary
        """
        metadata = {"filename": filename}
        
        # Extract Act name from title (first few lines)
        lines = text.split("\n")[:10]
        for line in lines:
            if "Act" in line and len(line.strip()) > 10:
                metadata["title"] = line.strip()
                break
        
        # Extract Act number
        act_num_match = re.search(
            r"Act\s+number\s+(\d+)\s+of\s+the\s+year\s+(\d+)",
            text,
            re.IGNORECASE
        )
        if act_num_match:
            metadata["act_number"] = act_num_match.group(1)
            metadata["year"] = act_num_match.group(2)
        
        # Extract date
        date_match = re.search(
            r"Date of (Authentication|Publication)[^\n]*\n([^\n]+)",
            text,
            re.IGNORECASE
        )
        if date_match:
            metadata["date"] = date_match.group(2).strip()
        
        return metadata
    
    def chunk_by_paragraphs(
        self, 
        text: str, 
        document_name: str = "unknown"
    ) -> List[TextChunk]:
        """
        Alternative chunking strategy that preserves paragraph boundaries.
        
        Args:
            text: The text to chunk
            document_name: Name of the source document
            
        Returns:
            List of TextChunk objects
        """
        text = self._clean_text(text)
        text = self._fix_words(text)
        metadata = self._extract_metadata(text, document_name)
        
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            # If adding this paragraph exceeds chunk_size, save current chunk
            if current_size + para_size > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunk = TextChunk(
                    text=chunk_text,
                    index=chunk_index,
                    start_char=0,
                    end_char=len(chunk_text),
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                        "char_count": len(chunk_text)
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
                current_chunk = []
                current_size = 0
            
            current_chunk.append(para)
            current_size += para_size + 2  # +2 for paragraph separator
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunk = TextChunk(
                    text=chunk_text,
                    index=chunk_index,
                    start_char=0,
                    end_char=len(chunk_text),
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                        "char_count": len(chunk_text)
                    }
                )
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} paragraph-based chunks from {document_name}")
        
        return chunks
