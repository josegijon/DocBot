from app.rag.retriever import retrieve
from app.rag.reranker import rerank

chunks = retrieve(
    "Donde se han realizado los analisis?", "2e38a119-6731-4e4a-b78a-32d578e93f0a"
)
print("Retrieved:", len(chunks))

top3 = rerank("Donde se han realizado los analisis?", chunks)
for c in top3:
    print(c["score"], c["page"], c["text"][:80])
