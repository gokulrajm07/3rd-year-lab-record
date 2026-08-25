[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KUMARAGURU-V-S/Lab-experiments/blob/main/GEN-AI-AND-LLM/Experiment-06-RAG-System-Vector-Databases/rag_system.ipynb)

# Experiment 06 - Retrieval-Augmented Generation (RAG) System Using Vector Databases

## Aim
To build a Retrieval-Augmented Generation (RAG) system that retrieves relevant document chunks from a vector database and uses an LLM to generate grounded answers.

## Objective
To understand how embeddings, vector similarity search, and LLM generation are combined to reduce hallucination and answer questions using an external knowledge base.

## Software Requirements
- Python 3.9 or above
- langchain, sentence-transformers, faiss-cpu libraries
- Hugging Face transformers (for the generator LLM)
- Jupyter Notebook / VS Code

## Theory
Retrieval-Augmented Generation (RAG) combines a retriever and a generator to answer queries using knowledge that may not be present in the LLM's parameters. Documents are split into chunks, converted into dense vector embeddings using a sentence-embedding model, and stored in a vector database (e.g., FAISS, Chroma, Pinecone).

At query time, the question is also embedded, and a similarity search (commonly cosine similarity or L2 distance) retrieves the top-k most relevant chunks. These retrieved chunks are inserted into the LLM's prompt as context, and the LLM generates an answer grounded in the retrieved evidence.

## Algorithm
1. Collect and split source documents into smaller text chunks.
2. Generate vector embeddings for each chunk using a sentence-transformer model.
3. Store the embeddings in a FAISS vector index.
4. Embed the incoming user query using the same embedding model.
5. Perform similarity search to retrieve the top-k most relevant chunks.
6. Construct a prompt combining the retrieved chunks and the query.
7. Pass the prompt to an LLM to generate the final grounded answer.

## How to Run

```bash
python rag_system.py
```

## Sample Input
```
query = "What is RAG in AI?"
```

## Sample Output
```
Retrieved Context: ['Retrieval-Augmented Generation combines document retrieval with text generation.', 'Vector databases store embeddings and support fast similarity search.']
Answer: RAG combines document retrieval with text generation using vector databases.
```

## Result
A Retrieval-Augmented Generation system was successfully implemented using FAISS as the vector database, producing answers grounded in retrieved document context.
