import re
from typing import List, Dict, Any, Optional, Tuple
from config import FINANCIAL_METRICS_SCHEMA

class FinancialMetricExtractor:
    """
    Extracts, standardizes, validates, and traces fundamental financial figures
    from SEC annual reports with mathematical integrity checks and exact citation anchors.
    """
    def __init__(self):
        self.schema = FINANCIAL_METRICS_SCHEMA

    def clean_num(self, val_str: str) -> Optional[float]:
        """Convert financial string to float with parenthetical negative support."""
        if not val_str:
            return None
        s = val_str.strip()
        is_neg = (s.startswith('(') and s.endswith(')')) or s.startswith('-')
        s = re.sub(r'[^\d\.]', '', s)
        if not s:
            return None
        try:
            val = float(s)
            return -val if is_neg else val
        except ValueError:
            return None

    def detect_magnitude(self, text: str) -> float:
        """Detect document-level unit magnitude (millions vs billions vs thousands)."""
        lower = text[:3000].lower()
        if "in billions" in lower or "(in billions" in lower:
            return 1e9
        elif "in millions" in lower or "(in millions" in lower or "amounts in millions" in lower:
            return 1e6
        elif "in thousands" in lower or "(in thousands" in lower:
            return 1e3
        return 1.0

    def extract_metrics_from_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts financial metrics defined in FINANCIAL_METRICS_SCHEMA from a single chunk.
        Enforces source traceability back to the chunk ID and exact text sentence.
        """
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        chunk_id = chunk.get("chunk_id", "")
        doc_year = metadata.get("fiscal_year", "2024")
        doc_section = metadata.get("section", "BODY")
        filename = metadata.get("filename", "")
        ticker = metadata.get("ticker", "GEN")

        doc_scale = self.detect_magnitude(text)
        extracted = []

        lines = text.split('\n')
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue

            line_lower = line_clean.lower()
            for metric_key, info in self.schema.items():
                # Check for synonym match in line
                matched_synonym = None
                for syn in info["synonyms"]:
                    # Exact phrase or word boundary match
                    if re.search(r'\b' + re.escape(syn) + r'\b', line_lower):
                        matched_synonym = syn
                        break

                if matched_synonym:
                    # Look for numerical figure in the line
                    # Patterns like "$383,285" or "383,285" or "(12,500)" or "4.25"
                    num_matches = re.findall(
                        r'(\(?\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:billion|million|b|m)?\)?)',
                        line_clean,
                        re.IGNORECASE
                    )
                    for raw_match in num_matches:
                        raw_str = raw_match.strip()
                        # Avoid single digits, years (like 2023, 2024), or section item numbers
                        if raw_str in ["2020", "2021", "2022", "2023", "2024", "2025", "2026", "1", "1A", "7", "8"]:
                            continue

                        val = self.clean_num(raw_str)
                        if val is None or abs(val) < 0.001:
                            continue

                        # Determine local multiplier
                        scale = doc_scale
                        lower_m = raw_str.lower()
                        if "billion" in lower_m or lower_m.endswith("b"):
                            scale = 1e9
                        elif "million" in lower_m or lower_m.endswith("m"):
                            scale = 1e6

                        # For EPS, scale is typically 1.0 (dollars per share)
                        if metric_key == "eps_diluted":
                            final_val = val
                            formatted = f"${val:.2f}"
                        else:
                            final_val = val * scale if scale > 1.0 and val < 1e7 else val
                            if abs(final_val) >= 1e9:
                                formatted = f"${final_val / 1e9:.2f}B"
                            elif abs(final_val) >= 1e6:
                                formatted = f"${final_val / 1e6:.2f}M"
                            else:
                                formatted = f"${final_val:,.2f}"

                        extracted.append({
                            "metric_key": metric_key,
                            "display_name": info["display_name"],
                            "category": info["category"],
                            "raw_value": final_val,
                            "formatted_value": formatted,
                            "unit": info["unit"],
                            "ticker": ticker,
                            "fiscal_year": doc_year,
                            "section": doc_section,
                            "filename": filename,
                            "chunk_id": chunk_id,
                            "evidence_quote": line_clean
                        })
                        break # Take primary matched figure per line

        return extracted

    def extract_from_corpus(self, chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Scans all chunks across corpus and groups extracted financial metrics
        by (ticker, fiscal_year, metric_key), keeping the most authoritative citation.
        """
        records = {}
        for c in chunks:
            items = self.extract_metrics_from_chunk(c)
            for item in items:
                key = (item["ticker"], item["fiscal_year"], item["metric_key"])
                # Prefer matches from Item 8 (Financial Statements) or Item 7 (MD&A)
                sec = item["section"].upper()
                priority = 3 if "ITEM_8" in sec or "ITEM 8" in sec else (2 if "ITEM_7" in sec or "ITEM 7" in sec else 1)
                
                if key not in records or priority > records[key]["priority"]:
                    item_copy = item.copy()
                    item_copy["priority"] = priority
                    records[key] = item_copy

        return records

    def validate_financial_consistency(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Mathematical sanity check engine:
        1. Gross Margin = (Revenue - COGS) / Revenue
        2. Operating Margin = Operating Income / Revenue
        3. Net Margin = Net Income / Revenue
        4. Free Cash Flow = Operating Cash Flow - CapEx
        """
        validations = []
        rev = metrics.get("revenue")
        cogs = metrics.get("cogs")
        gp = metrics.get("gross_profit")
        op_inc = metrics.get("operating_income")
        net_inc = metrics.get("net_income")
        ocf = metrics.get("operating_cash_flow")
        capex = metrics.get("capex")
        fcf = metrics.get("free_cash_flow")

        # 1. Gross Profit Check
        if rev and cogs and gp:
            calc_gp = rev - cogs
            diff = abs(calc_gp - gp)
            is_valid = (diff / max(abs(gp), 1.0)) < 0.05
            validations.append({
                "rule": "Gross Profit Integrity: Revenue - COGS == Gross Profit",
                "calculated": f"${calc_gp:,.2f}",
                "reported": f"${gp:,.2f}",
                "status": "VALID" if is_valid else "DISCREPANCY",
                "discrepancy_pct": round((diff / gp) * 100, 2)
            })

        # 2. Operating Margin Calculation
        if rev and op_inc and rev > 0:
            op_margin_pct = (op_inc / rev) * 100
            validations.append({
                "rule": "Operating Margin: Operating Income / Revenue",
                "calculated": f"{op_margin_pct:.2f}%",
                "status": "COMPUTED",
                "discrepancy_pct": 0.0
            })

        # 3. Net Margin Calculation
        if rev and net_inc and rev > 0:
            net_margin_pct = (net_inc / rev) * 100
            validations.append({
                "rule": "Net Profit Margin: Net Income / Revenue",
                "calculated": f"{net_margin_pct:.2f}%",
                "status": "COMPUTED",
                "discrepancy_pct": 0.0
            })

        # 4. Free Cash Flow Check
        if ocf and capex:
            calc_fcf = ocf - capex
            validations.append({
                "rule": "Free Cash Flow: Operating Cash Flow - CapEx",
                "calculated": f"${calc_fcf:,.2f}",
                "status": "COMPUTED" if not fcf else ("VALID" if abs(calc_fcf - fcf) < 1e6 else "DISCREPANCY"),
                "discrepancy_pct": 0.0
            })

        return validations

    def compute_temporal_comparison(
        self,
        metrics_t0: Dict[str, float],
        metrics_t1: Dict[str, float],
        year_t0: str,
        year_t1: str
    ) -> List[Dict[str, Any]]:
        """
        Computes Year-over-Year (YoY) dollar change and percentage growth rates.
        """
        comparison = []
        all_keys = set(metrics_t0.keys()) | set(metrics_t1.keys())

        for k in all_keys:
            v0 = metrics_t0.get(k)
            v1 = metrics_t1.get(k)
            info = self.schema.get(k, {"display_name": k, "unit": "USD"})

            if v0 is not None and v1 is not None:
                delta = v1 - v0
                pct_change = (delta / abs(v0)) * 100.0 if abs(v0) > 0 else 0.0
                comparison.append({
                    "metric_key": k,
                    "display_name": info["display_name"],
                    "value_t0": v0,
                    "value_t1": v1,
                    "delta_abs": round(delta, 2),
                    "growth_pct": round(pct_change, 2),
                    "trend": "Up" if delta > 0 else ("Down" if delta < 0 else "Flat")
                })

        return comparison
