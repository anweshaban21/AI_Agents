import requests
import pandas as pd
from dataCollect import build_financial_report
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
HEADERS = {
    "User-Agent": "Anwesha Banerjee banerjeeanwesha2002@gmail.com.com"
}

def build_vector_db(ticker: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    docs=build_financial_report(ticker) # Split the report preview into sections
    # docs = [
    #     "The stock market showed strong gains in Q3 driven by tech earnings.",
    #     "Federal Reserve maintains interest rates amid inflation concerns.",
    #     "Oil prices surge following OPEC production cut announcement.",
    #     "Cryptocurrency markets rally as institutional adoption increases.",
    #     "GDP growth forecast revised upward by IMF for emerging markets.",
    # ]
      # Return the query and the LLM's response as a dict
    return FAISS.from_texts(docs, embeddings)
#VECTOR_DB=build_vector_db("AAPL")
def report_to_chunks(report: dict) -> list[str]:
    chunks = []

    km = report.get("key_metrics", {})
    if km:
        chunks.append(
            f"Key metrics: Revenue {km.get('revenue', {}).get('val')}, "
            f"Net income {km.get('net_income', {}).get('val')}, "
            f"Assets {km.get('assets', {}).get('val')}, "
            f"Liabilities {km.get('liabilities', {}).get('val')}, "
            f"Operating cash flow {km.get('operating_cash_flow', {}).get('val')}, "
            f"Diluted EPS {km.get('eps', {}).get('val')}."
        )

    sections = report.get("sections", {})
    for name, text in sections.items():
        if text and len(text.strip()) > 30:
            # split large text into smaller pieces
            for i in range(0, len(text), 800):
                chunk = text[i:i+800].strip()
                if chunk:
                    chunks.append(f"{name}: {chunk}")

    preview = report.get("text_preview")
    if preview:
        for i in range(0, len(preview), 800):
            chunk = preview[i:i+800].strip()
            if chunk:
                chunks.append(f"preview: {chunk}")

    return chunks
res=build_financial_report("AAPL")
chunks=report_to_chunks(res)
print(chunks) # Print the first 3 chunks to verify