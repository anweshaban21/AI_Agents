import requests

HEADERS = {
    "User-Agent": "Anwesha Banerjee banerjeeanwesha2002@gmail.com.com"
}

import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any, List



# ---------- SEC helpers ----------

def ticker_to_cik(ticker: str) -> str:
    url = "https://www.sec.gov/files/company_tickers.json"
    data = requests.get(url, headers=HEADERS, timeout=30).json()
    ticker = ticker.upper()

    for _, item in data.items():
        if item["ticker"] == ticker:
            return str(item["cik_str"]).zfill(10)

    raise ValueError(f"Ticker not found: {ticker}")


def get_submissions(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def get_companyfacts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def latest_filing_info(submissions: dict, preferred_forms: List[str] = None) -> Dict[str, Any]:
    if preferred_forms is None:
        preferred_forms = ["10-Q", "10-K", "8-K"]

    recent = submissions["filings"]["recent"]
    df = pd.DataFrame(recent)

    for form in preferred_forms:
        hit = df[df["form"] == form]
        if not hit.empty:
            row = hit.iloc[0]
            return {
                "form": row["form"],
                "filingDate": row["filingDate"],
                "accessionNumber": row["accessionNumber"],
                "primaryDocument": row["primaryDocument"],
            }

    raise ValueError("No recent filing found for preferred forms.")


def filing_url(cik: str, accession_number: str, primary_document: str) -> str:
    accession_nodash = accession_number.replace("-", "")
    cik_nolead = str(int(cik))
    return f"https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{accession_nodash}/{primary_document}"


# ---------- XBRL extraction helpers ----------

def get_latest_usd_fact(companyfacts: dict, taxonomy_key: str) -> Optional[Dict[str, Any]]:
    """
    Example taxonomy_key:
    'RevenueFromContractWithCustomerExcludingAssessedTax'
    'NetIncomeLoss'
    'Assets'
    'Liabilities'
    'NetCashProvidedByUsedInOperatingActivities'
    """
    try:
        fact = companyfacts["facts"]["us-gaap"][taxonomy_key]
    except KeyError:
        return None

    units = fact.get("units", {})
    candidates = units.get("USD", []) or units.get("USD/shares", []) or []

    if not candidates:
        return None

    # Prefer annual/quarterly reported facts with an end date and filed date
    cleaned = [x for x in candidates if "val" in x and "end" in x]
    if not cleaned:
        return None

    cleaned.sort(key=lambda x: (x.get("end", ""), x.get("filed", "")), reverse=True)
    return cleaned[0]


def get_latest_eps(companyfacts: dict) -> Optional[Dict[str, Any]]:
    eps_keys = [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasicAndDiluted",
        "EarningsPerShareBasic",
    ]

    for key in eps_keys:
        try:
            fact = companyfacts["facts"]["us-gaap"][key]
            units = fact.get("units", {})
            candidates = units.get("USD/shares", [])
            cleaned = [x for x in candidates if "val" in x and "end" in x]
            if cleaned:
                cleaned.sort(key=lambda x: (x.get("end", ""), x.get("filed", "")), reverse=True)
                return {"taxonomy": key, **cleaned[0]}
        except KeyError:
            continue

    return None


def extract_key_metrics(companyfacts: dict) -> Dict[str, Any]:
    revenue_keys = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
    ]

    revenue = None
    for key in revenue_keys:
        revenue = get_latest_usd_fact(companyfacts, key)
        if revenue:
            revenue = {"taxonomy": key, **revenue}
            break

    metrics = {
        "revenue": revenue,
        "net_income": None,
        "assets": None,
        "liabilities": None,
        "operating_cash_flow": None,
        "eps": get_latest_eps(companyfacts),
    }

    ni = get_latest_usd_fact(companyfacts, "NetIncomeLoss")
    if ni:
        metrics["net_income"] = {"taxonomy": "NetIncomeLoss", **ni}

    assets = get_latest_usd_fact(companyfacts, "Assets")
    if assets:
        metrics["assets"] = {"taxonomy": "Assets", **assets}

    liabilities = get_latest_usd_fact(companyfacts, "Liabilities")
    if liabilities:
        metrics["liabilities"] = {"taxonomy": "Liabilities", **liabilities}

    ocf = get_latest_usd_fact(companyfacts, "NetCashProvidedByUsedInOperatingActivities")
    if ocf:
        metrics["operating_cash_flow"] = {
            "taxonomy": "NetCashProvidedByUsedInOperatingActivities",
            **ocf
        }

    return metrics


# ---------- Filing text extraction ----------

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_filing_text(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # remove scripts/styles
    for tag in soup(["script", "style", "ix:header", "header", "footer"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return clean_text(text)


def find_section(text: str, start_patterns: List[str], end_patterns: List[str], max_chars: int = 12000) -> str:
    lower = text.lower()

    start_idx = None
    for pat in start_patterns:
        m = re.search(pat, lower)
        if m:
            start_idx = m.start()
            break

    if start_idx is None:
        return ""

    end_idx = None
    search_from = lower[start_idx + 1:]

    for pat in end_patterns:
        m = re.search(pat, search_from)
        if m:
            end_idx = start_idx + 1 + m.start()
            break

    if end_idx is None:
        end_idx = min(len(text), start_idx + max_chars)

    return text[start_idx:end_idx].strip()


def extract_management_sections(full_text: str) -> Dict[str, str]:
    mda = find_section(
        full_text,
        start_patterns=[r"item 7\.*\s+management[’']?s discussion", r"management[’']?s discussion and analysis"],
        end_patterns=[r"item 7a\.", r"item 8\."]
    )

    risk_factors = find_section(
        full_text,
        start_patterns=[r"item 1a\.*\s+risk factors", r"risk factors"],
        end_patterns=[r"item 1b\.", r"item 2\."]
    )

    business = find_section(
        full_text,
        start_patterns=[r"item 1\.*\s+business", r"\bbusiness\b"],
        end_patterns=[r"item 1a\.", r"item 2\."]
    )

    return {
        "business": business[:12000],
        "risk_factors": risk_factors[:12000],
        "management_discussion": mda[:12000],
    }


# ---------- Final report builder ----------

def build_financial_report(ticker: str) -> Dict[str, Any]:
    cik = ticker_to_cik(ticker)
    submissions = get_submissions(cik)
    companyfacts = get_companyfacts(cik)

    filing = latest_filing_info(submissions, preferred_forms=["10-Q", "10-K", "8-K"])
    url = filing_url(cik, filing["accessionNumber"], filing["primaryDocument"])

    full_text = extract_filing_text(url)
    sections = extract_management_sections(full_text)
    metrics = extract_key_metrics(companyfacts)

    return {
        "ticker": ticker.upper(),
        "cik": cik,
        "entity_name": companyfacts.get("entityName"),
        "latest_filing": filing,
        "filing_url": url,
        "key_metrics": metrics,
        "sections": sections,
        "text_preview": full_text[:3000],
    }

#print(build_financial_report("AAPL"))
# ---------- Example ----------
#report = build_financial_report("AAPL")
#print(report)
# print("Company:", report["entity_name"])
# print("Latest filing:", report["latest_filing"])
# print("Filing URL:", report["filing_url"])
# print("\nRevenue:", report["key_metrics"]["revenue"])
# print("Net income:", report["key_metrics"]["net_income"])
# print("EPS:", report["key_metrics"]["eps"])
# print("\nMD&A preview:")
# print(report["sections"]["management_discussion"][:1500])