import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from fastapi import HTTPException

# PDF / DOCX parsing
from PyPDF2 import PdfReader
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------
# DATA MODELS
# -----------------------
@dataclass
class TextSegment:
    text: str
    page: int  # 1-indexed
    start: int
    end: int
    source_id: str = ""
    paragraph_index: int = 0
    chapter_title: str = ""
    type: str = "text"

# -----------------------
# PARSERS
# -----------------------
def extract_text_from_pdf(path: Path) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    """
    Returns:
        full_text: str
        source_map: List of (start_char_idx, end_char_idx, page_number, paragraph_index)
    """
    try:
        reader = PdfReader(str(path))
        full_text = ""
        source_map = []
        paragraph_counter = 0
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
                
            # Split into paragraphs (naive splitting by double newline)
            paragraphs = text.split('\n\n')
            
            for para in paragraphs:
                if not para.strip():
                    continue
                    
                # Add separator
                if full_text:
                    full_text += "\n\n"
                    
                start_idx = len(full_text)
                full_text += para
                end_idx = len(full_text)
                
                source_map.append((start_idx, end_idx, i + 1, paragraph_counter))
                paragraph_counter += 1
            
        return full_text, source_map
    except Exception as e:
        raise HTTPException(400, f"Error reading PDF: {e}")

def extract_text_from_docx(path: Path) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    try:
        doc = docx.Document(str(path))
        full_text = ""
        source_map = []
        paragraph_counter = 0
        
        for p in doc.paragraphs:
            text = p.text
            if not text.strip():
                continue
                
            if full_text:
                full_text += "\n\n"
                
            start_idx = len(full_text)
            full_text += text
            end_idx = len(full_text)
            
            # Treat as Page 1 for DOCX
            source_map.append((start_idx, end_idx, 1, paragraph_counter))
            paragraph_counter += 1
            
        return full_text, source_map
    except Exception as e:
        raise HTTPException(400, f"Error reading DOCX: {e}")

def extract_text_from_txt(path: Path) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        full_text = ""
        source_map = []
        paragraph_counter = 0
        
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if not para.strip():
                continue
                
            if full_text:
                full_text += "\n\n"
                
            start_idx = len(full_text)
            full_text += para
            end_idx = len(full_text)
            
            source_map.append((start_idx, end_idx, 1, paragraph_counter))
            paragraph_counter += 1
            
        return full_text, source_map
    except Exception as e:
        raise HTTPException(400, f"Error reading TXT: {e}")

# -----------------------
# CHAPTER + CODE DETECTION
# -----------------------
CHAPTER_RE = re.compile(r'(?im)(?:^|\n)((?:chapter|section)\s+\d+.*?)(?:\n|$)')

def detect_chapters(full_text: str) -> List[Tuple[int, str]]:
    chapters = []
    for m in CHAPTER_RE.finditer(full_text):
        title = m.group(1).strip()
        chapters.append((m.start(), title))
    
    if not chapters:
        return [(0, "Introduction")]
    return chapters

def extract_code_blocks(text: str) -> List[str]:
    code_blocks = []
    fenced = re.findall(r"```(?:[a-zA-Z0-9_\-]*)\n(.*?)```", text, re.DOTALL)
    inline = [m for m in re.findall(r"`([^`]+)`", text) if len(m) > 10]
    indented_candidates = re.findall(r"((?:\n(?: {4}|\t).+){2,})", text)
    code_indicators = {'{', '}', '=', '(', ')', 'def ', 'import ', 'return', 'var ', 'const ', 'function'}
    
    valid_indented = []
    for block in indented_candidates:
        if any(ind in block for ind in code_indicators):
            valid_indented.append(block.strip())

    for b in fenced + inline + valid_indented:
        if len(b.strip()) > 20:
            code_blocks.append(b.strip())
    return code_blocks

# -----------------------
# CHUNKING
# -----------------------
def chunk_with_metadata(text: str, source_map: List[Tuple[int, int, int, int]], chunk_size=2000, chunk_overlap=300) -> List[Dict[str, Any]]:
    """
    Splits text and assigns metadata including paragraph index.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chapters = detect_chapters(text)
    chunks = splitter.split_text(text)

    annotated = []
    last_pos = 0
    current_chapter_idx = 0

    for i, c in enumerate(chunks):
        start_pos = text.find(c, last_pos)
        if start_pos == -1:
            start_pos = text.find(c)
        
        end_pos = start_pos + len(c)
        last_pos = max(0, start_pos + len(c) - chunk_overlap)

        # Chapter
        while current_chapter_idx + 1 < len(chapters) and start_pos >= chapters[current_chapter_idx + 1][0]:
            current_chapter_idx += 1
        chapter_title = chapters[current_chapter_idx][1]
        
        # Resolve Page & Paragraph
        # We find the paragraph that contains the *center* of the chunk to avoid ambiguity,
        # or just the first overlapping paragraph.
        # Let's collect all overlapping paragraphs.
        
        pages = set()
        paragraphs = set()
        
        for p_start, p_end, p_num, para_idx in source_map:
            if max(start_pos, p_start) < min(end_pos, p_end):
                pages.add(p_num)
                paragraphs.add(para_idx)
        
        # Use the first paragraph index as the primary anchor for clustering
        primary_para_idx = min(paragraphs) if paragraphs else 0
        page_label = ", ".join(map(str, sorted(pages))) if pages else "Unknown"
        
        seg_type = "code" if extract_code_blocks(c) else "text"

        # Create Dataclass (we convert to dict for return to maintain compatibility with main.py)
        segment = TextSegment(
            text=c,
            page=min(pages) if pages else 1, # Primary page
            start=start_pos,
            end=end_pos,
            paragraph_index=primary_para_idx,
            chapter_title=chapter_title,
            type=seg_type
        )
        
        # Extra metadata for Chroma
        meta = asdict(segment)
        del meta["text"] # Text is stored separately in Chroma
        meta["page_label"] = page_label # Add string label
        
        annotated.append({
            "text": c,
            "metadata": meta
        })
        
    return annotated
