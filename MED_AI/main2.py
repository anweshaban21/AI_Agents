from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 🔑 API KEYS
PINECONE_API_KEY = "pcsk_31koiJ_G8DXngE8CLyR3baiDw2REp1smzMZyvu3iQH33WR3rnfFV3aAJh2Zbp2Kwd2BwYy"
GEMINI_API_KEY = "AIzaSyA3GBZcNz-YM6ncsJh8qjKhXP1Ur8Fw8wY"

# ✅ Gemini via OpenAI-compatible API
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 📦 Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "pdf-rag"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# 📌 Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 📄 Load PDF
#reader = PdfReader("random.pdf")
loader = PyPDFLoader("random.pdf")
documents = loader.load()
# text = ""
# for page in loader.pages:
#     text += page.extract_text()

# ✂️ Chunking
# def chunk_text(text, chunk_size=500, overlap=100):
#     chunks = []
#     for i in range(0, len(text), chunk_size - overlap):
#         chunks.append(text[i:i + chunk_size])
#     return chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

#chunks = chunk_text(text)

# 🔢 Store in Pinecone
vectors = []
for i, chunk in enumerate(chunks):
    vector = embed_model.encode(chunk.page_content).tolist()
    vectors.append({
        "id": str(i),
        "values": vector,
        "metadata": {"text": chunk.page_content}
    })


index.upsert(vectors=vectors)

print("✅ PDF stored!")

# 🔍 Query loop
while True:
    user_query = input("\nAsk a question (or type 'exit'): ")
    if user_query.lower() == "exit":
        break

    # Embed query
    query_vector = embed_model.encode(user_query).tolist()

    # Search Pinecone
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    context = "\n".join(
        [match["metadata"]["text"] for match in results["matches"]]
    )

    # 🧠 RAG prompt
    prompt = f"""
    Answer ONLY using the context below.

    Context:
    {context}

    Question:
    {user_query}

    Return answer in JSON format:
    {{"answer": "..."}}
    """

    # ✅ YOUR REQUIRED FORMAT
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\n💡 Answer:")
    print(response.choices[0].message.content)
