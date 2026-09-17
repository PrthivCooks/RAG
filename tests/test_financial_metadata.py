import pytest
from pathlib import Path
from utils.helpers import detect_sec_filing_metadata, segment_sec_sections, parse_financial_numbers

def test_detect_sec_filing_metadata():
    text = """
    COMPANY NAME: Apple Inc.
    TICKER: AAPL
    FORM TYPE: Form 10-K
    FISCAL YEAR: 2024
    PERIOD ENDED: September 28, 2024
    Item 1. Business
    """
    file_path = Path("AAPL_FY2024_10K.txt")
    meta = detect_sec_filing_metadata(text, file_path)

    assert meta["company"] == "Apple Inc."
    assert meta["ticker"] == "AAPL"
    assert meta["fiscal_year"] == "2024"
    assert meta["form_type"] == "Form 10-K"
    assert meta["is_financial_report"] is True

def test_segment_sec_sections():
    text = """
    Item 1. Business
    Apple Inc. designs, manufactures, and markets smartphones.

    Item 1A. Risk Factors
    The Company faces significant competition in all markets.

    Item 7. Management's Discussion and Analysis
    Total net sales were $391,035 million in fiscal 2024.

    Item 8. Consolidated Financial Statements
    Total revenue was $391,035 million.
    """
    sections = segment_sec_sections(text)
    codes = [s["code"] for s in sections]

    assert "Item 1" in codes
    assert "Item 1A" in codes
    assert "Item 7" in codes
    assert "Item 8" in codes

    # Ensure risk factor text is isolated
    sec_1a = next(s for s in sections if s["code"] == "Item 1A")
    assert "Item 1A" in sec_1a["text"]
    assert "significant competition" in sec_1a["text"]

def test_parse_financial_numbers():
    sample_text = "Net sales increased to $391,035 million with diluted EPS of $6.08 per share and operating margin of (12.4%)."
    parsed = parse_financial_numbers(sample_text)

    # Should detect 391035 million, 6.08, and -12.4%
    values = [p["parsed_value"] for p in parsed]
    assert any(abs(v - 3.91035e11) < 1e6 or abs(v - 391035.0) < 1.0 for v in values)
    assert any(abs(v - 6.08) < 0.01 for v in values)
    assert any(p["is_percentage"] for p in parsed)
