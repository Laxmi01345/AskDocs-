# mini implementation of RAG (Retrieval Augmented Generation) using sentence transformers and cerebras cloud sdk

from sentence_transformers import SentenceTransformer
import torch
import os
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
import numpy as np


load_dotenv() 
model = SentenceTransformer('all-MiniLM-L6-v2')

text = "The Amazon rainforest, often referred to as the lungs of the Earth, is home to an incredibly diverse range of species. It covers over 5.5 million square kilometers and spans across nine countries in South America. Deforestation has become a major concern, with large areas being cleared for agriculture and cattle ranching. This not only threatens biodiversity but also contributes significantly to global carbon emissions. Efforts are being made by governments and environmental organizations to protect this vital ecosystem through conservation programs and sustainable practices."
question = "What is the population of Brazil?"

chunks =""
fullchunks = []
for i in range(len(text)):
    if (text[i]!=".") :
        chunks += text[i]
    else :
        chunks+="." + "\n"
        fullchunks.append(chunks)
        chunks=""

       
chunk_embeddings = []
for i in range(len(fullchunks)):
    # print(fullchunks[i])
    embedding = model.encode(fullchunks[i])
    chunk_embeddings.append(embedding)

question_embedding = model.encode(question)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = []
for emb in chunk_embeddings:
    scores.append(cosine_sim(question_embedding, emb))

scores = np.array(scores)

for idx in scores:
    print(f"Score: {idx:.4f}")

top_indices = np.argsort(scores)[-3:][::-1]


context = ""
for idx in top_indices:
    context += fullchunks[idx]
    context += "\n\n"
    
# print("Top 3 relevant chunks:", context)



client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY"),
)

prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

chat_completion = client.chat.completions.create(
    model="llama-3.3-70b",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,
    max_tokens=300,
)

print(chat_completion.choices[0].message.content)
