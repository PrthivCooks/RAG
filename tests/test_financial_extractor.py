import pytest
from core.financial_extractor import FinancialMetricExtractor

def test_extract_metrics_from_chunk():
    extractor = FinancialMetricExtractor()
    chunk = {
        "chunk_id": "chunk-test-101",
        "text": (
            "Total net sales were $391,035 million in fiscal 2024. "
            "Operating income was $123,216 million, and net income reached $93,736 million. "
            "Diluted earnings per share was $6.08 per share."
        ),
        "metadata": {
            "ticker": "AAPL",
            "fiscal_year": "2024",
            "section": "Item 7",
            "filename": "AAPL_2024_10K.txt"
        }
    }

    metrics = extractor.extract_metrics_from_chunk(chunk)
    metric_keys = [m["metric_key"] for m in metrics]

    assert "revenue" in metric_keys
    assert "operating_income" in metric_keys
    assert "net_income" in metric_keys
    assert "eps_diluted" in metric_keys

    # Verify source traceability
    for m in metrics:
        assert m["chunk_id"] == "chunk-test-101"
        assert m["section"] == "Item 7"
        assert len(m["evidence_quote"]) > 10

def test_validate_financial_consistency():
    extractor = FinancialMetricExtractor()
    metrics = {
        "revenue": 391035.0,
        "cogs": 210352.0,
        "gross_profit": 180683.0,
        "operating_income": 123216.0,
        "net_income": 93736.0,
        "operating_cash_flow": 118254.0,
        "capex": 9451.0
    }

    validations = extractor.validate_financial_consistency(metrics)
    assert len(validations) >= 3

    # Check Gross profit validation
    gp_rule = next(v for v in validations if "Gross Profit Integrity" in v["rule"])
    assert gp_rule["status"] == "VALID"

    # Check Operating margin computed
    om_rule = next(v for v in validations if "Operating Margin" in v["rule"])
    assert "31.51%" in om_rule["calculated"]

def test_compute_temporal_comparison():
    extractor = FinancialMetricExtractor()
    m_2023 = {"revenue": 383285.0, "operating_income": 114301.0}
    m_2024 = {"revenue": 391035.0, "operating_income": 123216.0}

    comp = extractor.compute_temporal_comparison(m_2023, m_2024, "2023", "2024")
    rev_comp = next(c for c in comp if c["metric_key"] == "revenue")

    assert rev_comp["delta_abs"] == 7750.0
    assert round(rev_comp["growth_pct"], 1) == 2.0
    assert rev_comp["trend"] == "Up"
