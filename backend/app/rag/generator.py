from groq import Groq

from app.core.config import settings


client = Groq(api_key=settings.GROQ_API_KEY)


def generate(messages: list):

    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile", messages=messages, stream=True
    )

    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
