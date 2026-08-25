[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KUMARAGURU-V-S/Lab-experiments/blob/main/GEN-AI-AND-LLM/Experiment-02-Prompt-Engineering-Techniques/prompt_engineering.ipynb)

# Experiment 02 - Prompt Engineering Techniques for Content Generation, Reasoning and Task Automation

## Aim
To implement zero-shot, one-shot, few-shot, and chain-of-thought prompting techniques for content generation, reasoning, and task automation using a Large Language Model.

## Objective
To understand how the structure and content of a prompt influences the output quality of an LLM, and to practically apply different prompting strategies to solve content generation and reasoning tasks without modifying model weights.

## Software Requirements
- Python 3.9 or above
- OpenAI Python SDK / Hugging Face transformers
- Valid API key (if using OpenAI/Gemini) or a local pipeline model
- Jupyter Notebook / VS Code

## Theory
Prompt engineering is the practice of designing the input text (prompt) given to a Large Language Model so as to elicit the most accurate and useful response, without changing the model's parameters. It leverages the model's in-context learning ability.

Zero-shot prompting asks the model to perform a task with no examples, relying purely on instructions. One-shot and few-shot prompting provide one or more example input-output pairs within the prompt so the model can infer the expected pattern. Chain-of-Thought (CoT) prompting encourages the model to reason step-by-step before producing a final answer, which significantly improves performance on arithmetic, logical, and multi-step reasoning tasks.

## Algorithm
1. Define the task to be solved (e.g., arithmetic reasoning, content generation, classification).
2. Construct a zero-shot prompt containing only the instruction and query.
3. Construct a few-shot prompt containing 2–3 example input-output pairs followed by the query.
4. Construct a chain-of-thought prompt that asks the model to 'think step by step'.
5. Send each prompt to the LLM and record the response.
6. Compare the outputs of the three techniques for correctness and quality.

## How to Run

```bash
python prompt_engineering.py
```

## Sample Input
```
Sentiment classification query and a two-step arithmetic word problem
```

## Sample Output
```
=== Zero-shot ===
Sentiment: Positive

=== Few-shot ===
Review: 'The product quality is excellent!'
Sentiment: Positive

=== Chain-of-Thought ===
A: Let's think step by step. 120 - 45 = 75. 75 + 30 = 105. The answer is 105.
```

## Result
Zero-shot, few-shot, and chain-of-thought prompting techniques were successfully implemented, and their impact on content generation and reasoning accuracy was analysed and compared.
