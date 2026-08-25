[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KUMARAGURU-V-S/Lab-experiments/blob/main/GEN-AI-AND-LLM/Experiment-12-Deployment-and-Evaluation/deploy_eval.ipynb)

# Experiment 12 - Deployment and Evaluation of a Generative AI Application Using Cloud-Based APIs and AI Frameworks

## Aim
To deploy a Generative AI text-generation application as a web service using the Gradio framework, and to evaluate its output quality using standard NLP evaluation metrics.

## Objective
To understand the end-to-end lifecycle of a Generative AI application — from wrapping a model into a user-facing interface, to deploying it as a shareable web app, and quantitatively evaluating the quality of its generated output.

## Software Requirements
- Python 3.9 or above
- gradio, transformers, evaluate, rouge_score libraries
- OpenAI / Hugging Face Inference API (optional, for a cloud-hosted model)
- Jupyter Notebook / VS Code / Hugging Face Spaces for hosting

## Theory
Deploying a Generative AI application involves wrapping a trained/pre-trained model behind a user interface or API endpoint so that end-users can interact with it without needing to run code themselves. Frameworks such as Gradio and Streamlit allow a model inference function to be exposed as an interactive web application with just a few lines of code, and can be hosted on cloud platforms such as Hugging Face Spaces or connected to cloud LLM APIs.

Evaluating a Generative AI application is essential to ensure output quality before and after deployment. Common automatic evaluation metrics include ROUGE and BLEU (which measure n-gram overlap between generated text and a reference, commonly used for summarization/translation) and perplexity (which measures how well a language model predicts a sample). Together with qualitative human review, these metrics allow developers to monitor model performance, detect regressions, and compare different models/prompts objectively.

## Algorithm
1. Define an inference function that takes user text input and returns model-generated output.
2. Wrap the inference function using the Gradio Interface API with appropriate input/output components.
3. Launch the Gradio app locally (and optionally deploy to Hugging Face Spaces for cloud hosting).
4. Prepare a small evaluation set of generated outputs and corresponding reference texts.
5. Compute ROUGE scores between generated and reference texts using the `evaluate` library.
6. Report and interpret the evaluation metrics.

## How to Run

```bash
python deploy_eval.py
```

## Sample Input
```
Long article text submitted via the Gradio web interface + one generated/reference summary pair for evaluation
```

## Sample Output
```
Running on local URL: http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live
ROUGE Evaluation Scores: {'rouge1': 0.78, 'rouge2': 0.55, 'rougeL': 0.74, 'rougeLsum': 0.74}
```

## Result
A Generative AI text-summarization application was successfully deployed as a cloud-accessible web app using Gradio, and its output quality was evaluated using ROUGE metrics.
