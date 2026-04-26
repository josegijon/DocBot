from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, chunks: list):
    if not chunks:
        return []

    # Prepara los pares para el modelo
    pairs = [[query, chunk["text"]] for chunk in chunks]

    # Predice scores. Devuelve array de numpy con las puntuaciones
    scores = cross_encoder.predict(pairs)

    # Empareja chunks con su nuevos scores y ordena
    ranked_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    # Actualiza el score y devuelve el top 3
    final_top_3 = []
    for chunk, score in ranked_chunks[:3]:
        chunk["score"] = round(float(score), 4)
        final_top_3.append(chunk)

    return final_top_3
