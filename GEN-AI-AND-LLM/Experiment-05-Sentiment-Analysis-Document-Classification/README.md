[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KUMARAGURU-V-S/Lab-experiments/blob/main/GEN-AI-AND-LLM/Experiment-05-Sentiment-Analysis-Document-Classification/sentiment_classification.ipynb)

# Experiment 05 - Sentiment Analysis and Document Classification Using Foundation Models

## Aim
To perform sentiment analysis and multi-class document classification using pre-trained foundation models.

## Objective
To understand how foundation models fine-tuned for classification tasks, and zero-shot classification models based on natural language inference, can be applied to categorise text without task-specific training.

## Software Requirements
- Python 3.9 or above
- Hugging Face transformers library
- Pre-trained models: distilbert-base-uncased-finetuned-sst-2-english (sentiment), facebook/bart-large-mnli (zero-shot classification)
- Jupyter Notebook / VS Code

## Theory
Sentiment analysis classifies text according to the emotional tone expressed (e.g., Positive/Negative). Foundation models fine-tuned on labelled sentiment datasets such as SST-2 attach a classification head on top of a pre-trained transformer encoder to predict sentiment labels with associated confidence scores.

Document classification assigns a document to one or more predefined categories. Zero-shot classification models (e.g., BART-large-MNLI) reformulate classification as a Natural Language Inference (NLI) problem — the document is treated as a 'premise' and each candidate label is turned into a 'hypothesis' (e.g., 'This text is about {label}'); the model outputs the entailment probability for every label, enabling classification into arbitrary, previously unseen categories without any additional training.

## Algorithm
1. Load the sentiment-analysis pipeline with a fine-tuned foundation model.
2. Pass sample review sentences to the pipeline and record sentiment + score.
3. Load the zero-shot-classification pipeline with BART-large-MNLI.
4. Define a document and a list of candidate category labels.
5. Run zero-shot classification to obtain the probability for each label.
6. Display sentiment results and the top predicted document category.

## How to Run

```bash
python sentiment_classification.py
```

## Sample Input
```
Two product reviews + one news sentence with candidate labels [Politics, Economy, Sports, Technology]
```

## Sample Output
```
Review: The new smartphone has an amazing camera and battery life!
-> POSITIVE (0.999)

Review: The delivery was late and the packaging was damaged.
-> NEGATIVE (0.998)

Document: The central bank raised interest rates to control rising inflation.
Economy: 0.94
Politics: 0.04
Technology: 0.01
Sports: 0.01
```

## Result
Sentiment analysis and zero-shot document classification were successfully performed using pre-trained foundation models, correctly identifying sentiment polarity and document category.
