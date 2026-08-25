"""
Experiment 06: Retrieval-Augmented Generation (RAG) System Using Vector Databases
Course: CS4V48 - GenAI & LLM Laboratory
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def main():
    documents = [
        "The Eiffel Tower is located in Paris, France and was completed in 1889.",
        "Retrieval-Augmented Generation combines document retrieval with text generation.",
        "Python is a popular high-level programming language used in AI development.",
        "Vector databases store embeddings and support fast similarity search."
    ]

    print("Embedding documents...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = embed_model.encode(documents)

    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(doc_embeddings))

    query = "What is RAG in AI?"
    print(f"\nQuery: '{query}'")
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding), k=2)

    retrieved_chunks = [documents[i] for i in I[0]]
    print("Retrieved Context Chunks:", retrieved_chunks)

    context = " ".join(retrieved_chunks)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    
    print("\nGenerating grounded answer using Flan-T5...")
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    answer = generator(prompt, max_length=60)

    print("\nGenerated Answer:")
    print(answer[0]["generated_text"])

if __name__ == "__main__":
    main()
