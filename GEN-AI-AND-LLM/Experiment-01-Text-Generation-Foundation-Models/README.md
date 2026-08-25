[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KUMARAGURU-V-S/Lab-experiments/blob/main/GEN-AI-AND-LLM/Experiment-01-Text-Generation-Foundation-Models/text_generation.ipynb)

# Experiment 01 - Text Generation Using Pre-Trained Foundation Models

## Aim
To develop a text generation application using a pre-trained foundation model (GPT-2) with the Hugging Face Transformers library.

## Objective
To understand how pre-trained foundation models generate coherent text from a given prompt, and to explore decoding strategies such as greedy search, sampling, and top-k / top-p (nucleus) sampling that control the quality and diversity of generated text.

## Software Requirements
- Python 3.9 or above
- Hugging Face transformers library (`pip install transformers`)
- PyTorch (`pip install torch`)
- Google Colab / Jupyter Notebook / VS Code

## Theory
A foundation model is a large neural network pre-trained on massive amounts of text data in a self-supervised manner, which can then be adapted to a wide range of downstream tasks. GPT-2 (Generative Pre-trained Transformer 2) is a decoder-only transformer model trained to predict the next token in a sequence given all previous tokens.

Text generation works by feeding a prompt (seed text) into the model, which computes a probability distribution over the vocabulary for the next token. The next token is chosen using a decoding strategy — greedy decoding always picks the highest probability token, while sampling-based methods (temperature, top-k, top-p) introduce controlled randomness to make the generated text more natural and diverse.

The Hugging Face `pipeline` API abstracts away tokenization, model loading, and decoding, allowing text generation to be performed with just a few lines of code.

## Algorithm
1. Install and import the `transformers` library.
2. Load a pre-trained text-generation pipeline with a chosen foundation model (e.g., GPT-2).
3. Provide an input prompt string.
4. Set generation parameters — `max_length`, `num_return_sequences`, `temperature`, `top_k`, `top_p`.
5. Call the pipeline on the prompt to generate text.
6. Display and analyse the generated output.

## How to Run

```bash
python text_generation.py
```

## Sample Input
```
prompt = "Artificial Intelligence will transform the future of"
```

## Sample Output
```
--- Generated Text 1 ---
Artificial Intelligence will transform the future of healthcare, education, and transportation by enabling smarter decision making and automating repetitive tasks across industries.

--- Generated Text 2 ---
Artificial Intelligence will transform the future of work by creating new job roles while automating routine processes in manufacturing and services.
```

## Result
A text generation application using the pre-trained GPT-2 foundation model was successfully developed, and coherent text was generated from a given prompt using sampling-based decoding strategies.
