import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from fastapi import HTTPException

# PDF / DOCX parsing
from PyPDF2 import PdfReader
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------
# PARSERS
# -----------------------
def extract_text_from_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        return "\n\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception as e:
        raise HTTPException(400, f"Error reading PDF: {e}")

def extract_text_from_docx(path: Path) -> str:
    try:
        doc = docx.Document(str(path))
        return "\n\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        raise HTTPException(400, f"Error reading DOCX: {e}")

def extract_text_from_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(400, f"Error reading TXT: {e}")

# -----------------------
# CHAPTER + CODE DETECTION
# -----------------------
# Improved Regex: Captures "Chapter 1: Introduction" (rest of line)
CHAPTER_RE = re.compile(r'(?im)(?:^|\n)((?:chapter|section)\s+\d+.*?)(?:\n|$)')

def detect_chapters(full_text: str) -> List[Tuple[int, str]]:
    """Return list of (start_index, chapter_title) for each detected chapter."""
    chapters = []
    for m in CHAPTER_RE.finditer(full_text):
        title = m.group(1).strip()
        chapters.append((m.start(), title))
    
    if not chapters:
        return [(0, "Introduction")]
    return chapters

def extract_code_blocks(text: str) -> List[str]:
    """Catch fenced and inline code blocks with stricter heuristics."""
    code_blocks = []
    
    # 1. Fenced blocks (standard)
    fenced = re.findall(r"```(?:[a-zA-Z0-9_\-]*)\n(.*?)```", text, re.DOTALL)
    
    # 2. Inline code (backticks) - only if significant length
    inline = [m for m in re.findall(r"`([^`]+)`", text) if len(m) > 10]
    
    # 3. Indented blocks - Stricter rules
    # Must be indented by 4 spaces/tab, span >1 line, and look like code
    indented_candidates = re.findall(r"((?:\n(?: {4}|\t).+){2,})", text)
    
    code_indicators = {'{', '}', '=', '(', ')', 'def ', 'import ', 'return', 'var ', 'const ', 'function'}
    
    valid_indented = []
    for block in indented_candidates:
        # Check for code-like characters
        if any(ind in block for ind in code_indicators):
            valid_indented.append(block.strip())

    for b in fenced + inline + valid_indented:
        if len(b.strip()) > 20:
            code_blocks.append(b.strip())
            
    return code_blocks

# -----------------------
# CHUNKING
# -----------------------
def chunk_with_metadata(text: str, chunk_size=2000, chunk_overlap=300) -> List[Dict[str, Any]]:
    """
    Splits text into chunks and assigns metadata (Chapter Title, Type).
    Uses a sliding window to accurately track chunk location.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chapters = detect_chapters(text)
    chunks = splitter.split_text(text)

    annotated = []
    last_pos = 0  # Sliding pointer to track position
    current_chapter_idx = 0

    for c in chunks:
        # Find chunk starting from last known position to handle duplicates
        start_pos = text.find(c, last_pos)
        if start_pos == -1:
            # Fallback: if text normalization changed something, reset search (rare)
            start_pos = text.find(c)
        
        # Update pointer (allow overlap for next search)
        last_pos = max(0, start_pos + len(c) - chunk_overlap)

        # Determine Chapter
        # Advance chapter index if we've passed the start of the next chapter
        while current_chapter_idx + 1 < len(chapters) and start_pos >= chapters[current_chapter_idx + 1][0]:
            current_chapter_idx += 1
        
        chapter_title = chapters[current_chapter_idx][1]
        
        metadata = {
            "chapter_title": chapter_title, 
            "type": "text"
        }
        
        if extract_code_blocks(c):
            metadata["type"] = "code"

        annotated.append({
            "text": c,
            "metadata": metadata
        })
        
    return annotated
