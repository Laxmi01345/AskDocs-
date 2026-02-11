import os
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

load_dotenv()

_client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))


def answer_question(context: str, question: str):
    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
    chat_completion = _client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300,
    )
    return chat_completion.choices[0].message.content