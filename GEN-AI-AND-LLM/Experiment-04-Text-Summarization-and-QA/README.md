[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KUMARAGURU-V-S/Lab-experiments/blob/main/GEN-AI-AND-LLM/Experiment-04-Text-Summarization-and-QA/summarization_qa.ipynb)

# Experiment 04 - Text Summarization and Question-Answering System Using Large Language Models

## Aim
To develop a text summarization system and a question-answering system using pre-trained Large Language Models (BART and DistilBERT).

## Objective
To understand abstractive summarization and extractive question-answering pipelines built on transformer encoder-decoder and encoder-only architectures respectively.

## Software Requirements
- Python 3.9 or above
- Hugging Face transformers library
- Pre-trained models: facebook/bart-large-cnn (summarization), distilbert-base-cased-distilled-squad (QA)
- Jupyter Notebook / VS Code

## Theory
Text summarization condenses a long passage into a shorter version while preserving its key information. Abstractive summarization models such as BART use an encoder-decoder transformer architecture: the encoder reads the full document and the decoder generates a new, fluent summary rather than merely extracting sentences.

Question-Answering (QA) systems extract or generate an answer to a natural-language question from a given context passage. Extractive QA models such as DistilBERT (fine-tuned on the SQuAD dataset) predict the start and end token positions of the answer span within the context passage using a classification head placed over the encoder outputs.

## Algorithm
1. Load the summarization pipeline with a pre-trained BART model.
2. Provide a long passage of text and generate a summary with defined min/max length.
3. Load the question-answering pipeline with a pre-trained DistilBERT-SQuAD model.
4. Provide a context passage and a natural-language question.
5. Run the QA pipeline to extract the answer span along with a confidence score.
6. Display the summary and the answer.

## How to Run

```bash
python summarization_qa.py
```

## Sample Input
```
Article about Generative AI & LLMs + Question: 'What are Large Language Models trained on?'
```

## Sample Output
```
Summary:
Generative AI models produce new content such as text, images, audio and video. Large Language Models are trained on massive text corpora and perform many NLP tasks.

Question: What are Large Language Models trained on?
Answer: massive text corpora | Confidence: 0.87
```

## Result
A text summarization system using BART and a question-answering system using DistilBERT-SQuAD were successfully developed and tested on sample text.
