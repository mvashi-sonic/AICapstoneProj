import faiss
import pickle
from sentence_transformers import SentenceTransformer

INDEX = faiss.read_index("./faiss/financial_reports.index")
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

with open("./faiss/financial_reports_metadata.pkl", "rb") as f:
    METADATA = pickle.load(f)
#INDEX = faiss.IndexFlatIP(384)
#INDEX.add(embeddings)

def retrieve(question):

    # embed question
    query_embedding = embedder.encode(
        [question],
        convert_to_numpy=True  # ,
        # normalize_embeddings=True
    )

    print(query_embedding.shape)
    print(query_embedding[0][:10])

    # search FAISS
    scores, ids = INDEX.search(
        query_embedding,
        k=50
    )
    print(scores)
    print(ids)
    # return chunks
    records = []
    for score, idx in zip(scores[0], ids[0]):
        chunk = METADATA[idx]
        dict = {}
        dict["score"] = f"{score:.4f}"
        dict["ticker"] = chunk["ticker"]
        dict["section"] = chunk["section"]
        dict["year"] = chunk["section"]
        dict["reference"] = chunk["reference"]
        records.append(dict)
    return records