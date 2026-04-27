from app.rag.prompt_builder import build_prompt


def test_chat():

    chunks = [{"text": "donde se hacen los analisis", "page": 1, "score": 0.9}]
    history = []
    query = "Donde se realiza el analisis"

    result = build_prompt(query, chunks, history)

    assert isinstance(result, list)
    assert result[0]["role"] == "system"
    assert result[-1]["role"] == "user"
    assert result[-1]["content"] == query
