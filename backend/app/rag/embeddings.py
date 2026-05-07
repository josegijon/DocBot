def create_embeddings(texts, embeddings_model):
    """
    Transforma una lista de textos en sus representaciones vectoriales (embeddings).

    Args:
        texts (list[str]): Textos a procesar.
        embeddings_model: Modelo de SentenceTransformers cargado.

    Returns:
        list[list[float]]: Lista de vectores numéricos.
    """
    return embeddings_model.encode(texts).tolist()