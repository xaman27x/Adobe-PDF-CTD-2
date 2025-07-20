import torch
from torch.nn.functional import cosine_similarity

def rank_chunks(persona, job, chunks, embedder, top_k=5):
    query = f"{persona} needs to: {job}"
    query_embedding = embedder.encode([query])
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embedder.encode(chunk_texts)

    scores = cosine_similarity(query_embedding, chunk_embeddings).squeeze(0)
    top_indices = torch.topk(scores, top_k).indices.tolist()

    ranked = []
    for rank, idx in enumerate(top_indices, 1):
        chunk = chunks[idx]
        chunk["importance_rank"] = rank
        chunk["score"] = float(scores[idx])
        ranked.append(chunk)
    return ranked
