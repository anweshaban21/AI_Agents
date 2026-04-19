import streamlit as st
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
import operator
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
import yfinance as yf
from dotenv import load_dotenv
load_dotenv(override=True)  
import os
from dataCollect import build_financial_report
import json
from company import detect_company_with_llm


class AgentState(TypedDict):
    query:str
    messages: Annotated[list[BaseMessage], operator.add]




from langchain_core.documents import Document

def build_vector_db(ticker: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    report = build_financial_report(ticker)
    docs = []

    km = report.get("key_metrics", {})
    if km:
        docs.append(
            Document(
                page_content=(
                    f"Revenue {km.get('revenue', {}).get('val')}, "
                    f"Net income {km.get('net_income', {}).get('val')}, "
                    f"EPS {km.get('eps', {}).get('val')}, "
                    f"Operating cash flow {km.get('operating_cash_flow', {}).get('val')}"
                ),
                metadata={"section": "key_metrics", "ticker": ticker}
            )
        )

    for sec_name, sec_text in report.get("sections", {}).items():
        if sec_text:
            for i in range(0, len(sec_text), 800):
                chunk = sec_text[i:i+800].strip()
                if chunk:
                    docs.append(
                        Document(
                            page_content=chunk,
                            metadata={"section": sec_name, "ticker": ticker}
                        )
                    )

    return FAISS.from_documents(docs, embeddings)









@tool
def finance_api(ticker: str) -> str:
    """Fetch live finance data like price, volume, high, low for a given stock ticker."""
    try:
        stock = yf.Ticker(ticker)

        info = stock.info
        hist = stock.history(period="1d")

        if hist.empty:
            return f"No market data found for {ticker}"

        latest = hist.iloc[-1]

        result = f"""
📊 Stock Data for {ticker.upper()}

Current Price: {info.get('currentPrice')}
Previous Close: {info.get('previousClose')}

Day High: {latest['High']}
Day Low: {latest['Low']}

Volume: {latest['Volume']}

Market Cap: {info.get('marketCap')}
52 Week High: {info.get('fiftyTwoWeekHigh')}
52 Week Low: {info.get('fiftyTwoWeekLow')}
"""

        return result.strip()

    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"
@tool
def news_api(query: str) -> str:
    """Fetch latest news headlines for a given topic."""
    return f"[News API] Mock headlines for: {query}"

@tool
def volatility_check(ticker: str) -> str:
    """Calculate volatility using standard deviation of price deviation from SMA."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")

        if hist.empty:
            return f"No data found for {ticker}"

        
        hist["SMA_20"] = hist["Close"].rolling(window=20).mean()

       
        hist = hist.dropna()

        
        hist["deviation"] = (hist["Close"] - hist["SMA_20"]) / hist["SMA_20"]

        
        volatility = hist["deviation"].std()

        return f"""
📉 SMA-Based Volatility for {ticker.upper()}

Volatility (std of deviation from SMA-20): {volatility:.2%}

Interpretation:
- < 2% → Very stable (price sticks to trend)
- 2% - 5% → Moderate volatility
- > 5% → High volatility (price swings far from trend)
""".strip()

    except Exception as e:
        return f"Error calculating volatility for {ticker}: {str(e)}"

@tool
def vector_db_search(ticker: str, query: str) -> str:
    """Search the vector database for relevant financial report sections."""
    vector_db = build_vector_db(ticker)
    docs = vector_db.similarity_search(query, k=3)

    results = []
    for d in docs:
        results.append(f"[{d.metadata.get('section')}] {d.page_content}")

    return "\n\n".join(results)

TOOLS = [finance_api, news_api, volatility_check, vector_db_search]




llm = ChatAnthropic(                                                
    model="claude-haiku-4-5-20251001",                              
    api_key=os.getenv("CLAUDE_API_KEY"),
    temperature=0,
).bind_tools(TOOLS)

# ── Nodes ──────────────────────────────────────────────────────────────────────

def llm_invoking_node(state: AgentState) -> AgentState:
    """Calls the LLM; it decides whether to call a tool or respond directly."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(TOOLS)




graph = StateGraph(AgentState)

graph.add_node("llm_invoking_node", llm_invoking_node)
graph.add_node("tool_node",         tool_node)

graph.set_entry_point("llm_invoking_node")

graph.add_conditional_edges(
    "llm_invoking_node",
    tools_condition,
    {"tools": "tool_node", END: END},  
)
graph.add_edge("tool_node", "llm_invoking_node")  

app = graph.compile()



if __name__ == "__main__":
    st.set_page_config(page_title="Financial Assistant", layout="centered")

    st.title("📊 Financial Report Assistant")

    st.write("Enter a query like:")
    st.code("Give me report of Apple")

    query = st.text_input("Enter your query:")
    ticker = detect_company_with_llm(query).get("ticker")
    if st.button("Get Result"):
        if query:
            with st.spinner("Detecting company..."):
                result = app.invoke({"messages": [HumanMessage(content=query)]})

        # Display result
            if result:
                st.success("Company Detected ✅")

                st.write("Answer:", result["messages"][-1].content)
                

            else:
                st.error("Could not detect company ❌")

        else:
            st.warning("Please enter a query")

    
    # result = app.invoke({"messages": [HumanMessage(content=query)]})
    # print("Answer:", result["messages"][-1].content)

