import os
import re
from pathlib import Path
from datetime import datetime
def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text content from a PDF file using pypdf."""
    text_content = []
    try:
        import pypdf
        reader = pypdf.PdfReader(str(file_path))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
    except ImportError:
        print(f"pypdf not installed. Please install pypdf to process PDF files: {file_path}")
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return "\n".join(text_content)

def extract_text_from_docx(file_path: Path) -> str:
    """Extract text content from a DOCX file using python-docx."""
    text_content = []
    try:
        import docx
        doc = docx.Document(str(file_path))
        for para in doc.paragraphs:
            if para.text.strip():
                text_content.append(para.text)
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    text_content.append(" | ".join(row_text))
    except ImportError:
        print(f"python-docx not installed. Please install python-docx to process DOCX files: {file_path}")
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return "\n".join(text_content)

def extract_text_from_txt(file_path: Path) -> str:
    """Extract text content from a plain TXT file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        return ""

def clean_extracted_text(text: str) -> str:
    """Normalize whitespace and merge single newlines within paragraphs."""
    if not text:
        return ""
    # Normalize carriage returns and horizontal whitespace
    text = text.replace('\r', '')
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Rebuild paragraphs to merge layout-induced line wraps
    lines = text.split('\n')
    paragraphs = []
    current_para = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
        else:
            # Merge hyphenated words across lines
            if stripped.endswith('-') and len(stripped) > 1:
                current_para.append(stripped[:-1])
            else:
                current_para.append(stripped)
                
    if current_para:
        paragraphs.append(" ".join(current_para))
        
    cleaned = "\n\n".join(paragraphs)
    return re.sub(r' +', ' ', cleaned)

def extract_document_text(file_path: Path) -> str:
    """Detect file type and extract text content."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif suffix == ".docx":
        text = extract_text_from_docx(file_path)
    elif suffix == ".txt":
        text = extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file format: {suffix} for {file_path}")
        return ""
        
    return clean_extracted_text(text)

def detect_department(text: str, file_path: Path) -> str:
    """
    Heuristically detect department from text content or path names.
    Looks for department names in the file hierarchy first, then checks text.
    """
    # Check parent directory name first (e.g. data/HR/policy.txt -> HR)
    parent_name = file_path.parent.name
    if parent_name.lower() not in ["data", "rag", "."]:
        return parent_name.capitalize()

    # Scan first 2000 characters for department hints
    snippet = text[:2000].lower()
    
    # Check for direct declarations, e.g., "Department: Human Resources"
    match = re.search(r"(?:department|dept\.|team):\s*([a-zA-Z\s\-]+)", snippet)
    if match:
        dept = match.group(1).strip().title()
        if len(dept) < 30:  # sanity check length
            return dept
            
    # Keywords mapping
    keywords = {
        "Human Resources": ["hr", "human resources", "onboarding", "employee", "leave policy", "vacation", "recruiting", "benefits"],
        "Finance": ["finance", "refund", "billing", "invoice", "payment", "revenue", "audit", "budget", "tax"],
        "Legal": ["legal", "nda", "contract", "agreement", "compliance", "terms of service", "privacy policy", "litigation"],
        "Engineering": ["engineering", "software", "git", "code", "architecture", "deployment", "database", "api", "release notes"],
        "Operations": ["operations", "ops", "logistics", "vendor", "facility", "inventory", "procurement", "supply chain"],
        "Security": ["security", "firewall", "cybersecurity", "password", "gdpr", "data retention", "access control", "iso 27001"],
        "Marketing": ["marketing", "seo", "campaign", "social media", "brand", "advertising", "pr"]
    }
    
    for dept, terms in keywords.items():
        for term in terms:
            if re.search(r"\b" + re.escape(term) + r"\b", snippet):
                return dept
                
    return "General"

def detect_sec_filing_metadata(text: str, file_path: Path) -> dict:
    """
    Heuristically detect SEC filing metadata: Company, Ticker, Fiscal Year, Period, Form Type.
    Combines filename cues and header text inspection.
    """
    fname = file_path.name.upper()
    snippet = text[:4000]
    
    # 1. Detect Ticker and Company Name
    ticker = None
    company = None
    
    known_companies = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "GOOG": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "NVDA": "NVIDIA Corporation",
        "META": "Meta Platforms Inc.",
        "TSLA": "Tesla Inc.",
        "JPM": "JPMorgan Chase & Co."
    }
    
    # Check filename first (e.g. AAPL_2024_10K.txt)
    for tick, comp in known_companies.items():
        if re.search(r'\b' + re.escape(tick) + r'\b', fname):
            ticker = tick
            company = comp
            break

    # If not found in filename, scan header text
    if not ticker:
        for tick, comp in known_companies.items():
            if re.search(r'\b' + re.escape(tick) + r'\b', snippet) or comp.lower() in snippet.lower():
                ticker = tick
                company = comp
                break
                
    # Fallback company detection via SEC header pattern
    if not company:
        comp_match = re.search(r"(?:exact name of registrant as specified in its charter|company name):\s*([A-Za-z0-9\s,\.\-]+)", snippet, re.IGNORECASE)
        if comp_match:
            candidate = comp_match.group(1).strip()
            if len(candidate) < 60:
                company = candidate.title()
                
    # 2. Detect Fiscal Year
    fiscal_year = None
    # Check filename for 20XX
    year_match = re.search(r'\b(20[12][0-9])\b', fname)
    if year_match:
        fiscal_year = year_match.group(1)
    else:
        # Check text (e.g., "for the fiscal year ended December 31, 2024" or "fiscal year 2024")
        text_year_match = re.search(r"(?:fiscal year ended|fiscal year|for the period ended)[^\n\r\d]{0,40}(20[12][0-9])", snippet, re.IGNORECASE)
        if text_year_match:
            fiscal_year = text_year_match.group(1)
        else:
            # Fallback to any prominent 20XX year in first 1500 chars
            fallback_year = re.search(r'\b(20[12][0-9])\b', snippet[:1500])
            if fallback_year:
                fiscal_year = fallback_year.group(1)

    # 3. Detect Form Type and Period
    form_type = "Annual Report"
    period = "FY"
    if "10-K" in fname or "10K" in fname or "form 10-k" in snippet.lower():
        form_type = "Form 10-K"
        period = "FY"
    elif "10-Q" in fname or "10Q" in fname or "form 10-q" in snippet.lower():
        form_type = "Form 10-Q"
        q_match = re.search(r'\b(Q[1-4])\b', fname) or re.search(r'(first|second|third|fourth)\s+quarter', snippet, re.IGNORECASE)
        if q_match:
            period = q_match.group(1).upper()
        else:
            period = "Q"

    is_financial = bool(ticker or "10-K" in form_type or "annual report" in snippet.lower() or "consolidated financial statements" in snippet.lower())

    return {
        "company": company or "Unknown Company",
        "ticker": ticker or "GEN",
        "fiscal_year": fiscal_year or "2024",
        "reporting_period": period,
        "form_type": form_type,
        "is_financial_report": is_financial
    }

def segment_sec_sections(text: str) -> list:
    """
    Segments corporate filing text into structured SEC Item sections.
    Returns list of dicts: [{'code': 'ITEM_1A', 'title': 'Risk Factors', 'start': int, 'end': int, 'text': str}]
    """
    from config import SEC_SECTIONS
    
    # Compile regex pattern for identifying Item headers
    # E.g. "Item 1A. Risk Factors", "ITEM 7 - Management's Discussion", etc.
    markers = []
    for key, info in SEC_SECTIONS.items():
        code_str = info["code"]
        # Match 'Item 1A' with optional period, colon, dash or newline
        pat = r'(?:^|\n)\s*(?:' + re.escape(code_str) + r')[\.\:\-\s]+([^\n\r]{2,80})'
        for m in re.finditer(pat, text, re.IGNORECASE):
            markers.append({
                "section_key": key,
                "code": info["code"],
                "default_title": info["title"],
                "matched_title": m.group(1).strip(),
                "start": m.start()
            })
            
    if not markers:
        # No SEC item markers found, treat as single general section
        return [{
            "code": "GENERAL",
            "title": "General Content",
            "start": 0,
            "end": len(text),
            "text": text
        }]
        
    # Deduplicate same-code markers (e.g. Table of Contents vs actual body).
    # If the same Item code appears multiple times, choose the latest occurrence (body).
    code_to_markers = {}
    for m in markers:
        code = m["code"]
        if code not in code_to_markers:
            code_to_markers[code] = []
        code_to_markers[code].append(m)

    filtered_markers = []
    for code, m_list in code_to_markers.items():
        filtered_markers.append(m_list[-1])

    filtered_markers.sort(key=lambda x: x["start"])
        
    sections = []
    for i, marker in enumerate(filtered_markers):
        start = marker["start"]
        end = filtered_markers[i + 1]["start"] if i + 1 < len(filtered_markers) else len(text)
        sec_text = text[start:end].strip()
        
        sections.append({
            "code": marker["code"],
            "title": marker.get("default_title", marker.get("matched_title", "Section")),
            "start": start,
            "end": end,
            "text": sec_text
        })
        
    return sections

def parse_financial_numbers(text: str) -> list:
    """
    Extracts numerical and monetary figures with their scale and surrounding semantic context.
    E.g. '$383,285 million', '(12.4%)', '$4.25 per share'.
    """
    results = []
    # Pattern matching currencies, scales, and percentages
    pattern = re.compile(
        r'(\(?[\$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:(?:[mb]illion|[tbk]|%|percent))?\)?(?:\s*per\s+share)?)',
        re.IGNORECASE
    )
    
    for match in pattern.finditer(text):
        val_str = match.group(1).strip()
        # Filter trivial single digits without currency/units
        if re.match(r'^\d$', val_str):
            continue
            
        is_negative = val_str.startswith('(') and val_str.endswith(')')
        clean_num_str = re.sub(r'[^\d\.]', '', val_str)
        if not clean_num_str:
            continue
            
        try:
            num = float(clean_num_str)
            if is_negative:
                num = -num
                
            scale = 1.0
            lower = val_str.lower()
            if "billion" in lower or lower.endswith("b"):
                scale = 1e9
            elif "million" in lower or lower.endswith("m"):
                scale = 1e6
            elif "thousand" in lower or lower.endswith("k"):
                scale = 1e3
                
            results.append({
                "raw_text": val_str,
                "parsed_value": num * scale if scale > 1.0 else num,
                "scale": scale,
                "is_percentage": "%" in val_str or "percent" in lower,
                "start": match.start(),
                "end": match.end()
            })
        except ValueError:
            continue
            
    return results

def extract_metadata(file_path: Path, text: str) -> dict:
    """Extract metadata including filename, modified date, department, and SEC financial tags."""
    stats = file_path.stat()
    mod_time = datetime.fromtimestamp(stats.st_mtime)
    
    # Try to parse date from document text if possible (e.g., "Date: 2026-05-12")
    doc_date = None
    date_match = re.search(r"date:\s*([a-zA-Z0-9\s,\-]+)", text[:1000], re.IGNORECASE)
    if date_match:
        try:
            date_str = date_match.group(1).strip()
            date_str_clean = re.sub(r'\s+', ' ', date_str)
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%B %d, %Y", "%d %B %Y"):
                try:
                    doc_date = datetime.strptime(date_str_clean, fmt)
                    break
                except ValueError:
                    continue
        except Exception:
            pass
            
    final_date = doc_date if doc_date else mod_time

    # Financial SEC metadata extraction
    sec_meta = detect_sec_filing_metadata(text, file_path)

    # Department detection (if SEC financial report, automatically set department to Finance)
    detected_dept = "Finance" if sec_meta["is_financial_report"] else detect_department(text, file_path)

    return {
        "filename": file_path.name,
        "filepath": str(file_path.absolute().as_posix()),
        "last_modified": mod_time.strftime("%Y-%m-%d %H:%M:%S"),
        "doc_date": final_date.strftime("%Y-%m-%d"),
        "department": detected_dept,
        "file_size_bytes": stats.st_size,
        # Enhanced financial tags
        "company": sec_meta["company"],
        "ticker": sec_meta["ticker"],
        "fiscal_year": sec_meta["fiscal_year"],
        "reporting_period": sec_meta["reporting_period"],
        "form_type": sec_meta["form_type"],
        "is_financial_report": sec_meta["is_financial_report"]
    }

