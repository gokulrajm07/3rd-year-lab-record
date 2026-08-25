[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KUMARAGURU-V-S/Lab-experiments/blob/main/GEN-AI-AND-LLM/Experiment-10-Fine-Tuning-PreTrained-Model/finetune_lm.ipynb)

# Experiment 10 - Fine-Tuning a Pre-Trained Language Model for a Domain-Specific Application

## Aim
To fine-tune a pre-trained language model (DistilBERT) on a domain-specific labelled dataset for a text-classification task.

## Objective
To understand the process of transfer learning — adapting a general-purpose pre-trained model to a specific downstream task and domain by continuing training on a smaller, labelled dataset.

## Software Requirements
- Python 3.9 or above
- transformers, datasets, torch, scikit-learn libraries
- Pre-trained model: distilbert-base-uncased
- GPU-enabled environment recommended (Google Colab)

## Theory
Fine-tuning adapts a pre-trained foundation model to a specific downstream task by continuing training on a smaller, task/domain-specific labelled dataset, updating some or all of the model's weights. This leverages the general linguistic knowledge already captured during pre-training, requiring far less data and compute than training from scratch.

For text classification, a classification head (a linear layer) is added on top of the pre-trained transformer encoder's pooled output. During fine-tuning, cross-entropy loss between the predicted class probabilities and the true labels is minimised using gradient descent (typically with the AdamW optimizer and a small learning rate) over a few epochs.

## Algorithm
1. Load a domain-specific labelled dataset (text + label pairs).
2. Load the pre-trained tokenizer and tokenize the dataset (padding/truncation).
3. Load the pre-trained model with a classification head matching the number of labels.
4. Define training arguments (learning rate, epochs, batch size).
5. Train the model on the training split using the Hugging Face Trainer API.
6. Evaluate the fine-tuned model on the validation/test split.
7. Save the fine-tuned model for later inference.

## How to Run

```bash
python finetune_lm.py
```

## Sample Input
```
IMDB movie review dataset (2000 training samples, 500 test samples), 2 epochs fine-tuning
```

## Sample Output
```
Epoch 1/2 - loss: 0.41 - accuracy: 0.83
Epoch 2/2 - loss: 0.24 - accuracy: 0.89
Evaluation metrics: {'eval_loss': 0.28, 'eval_accuracy': 0.887}
```

## Result
The pre-trained DistilBERT model was successfully fine-tuned on a domain-specific sentiment-classification dataset, improving task-specific accuracy through transfer learning.
