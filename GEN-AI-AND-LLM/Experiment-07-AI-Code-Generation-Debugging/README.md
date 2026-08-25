[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KUMARAGURU-V-S/Lab-experiments/blob/main/GEN-AI-AND-LLM/Experiment-07-AI-Code-Generation-Debugging/code_assistant.ipynb)

# Experiment 07 - AI-Powered Code Generation and Debugging Assistant

## Aim
To develop an AI-powered assistant that generates code from natural-language descriptions and helps identify/fix bugs in existing code using a pre-trained code-generation model.

## Objective
To understand how transformer models trained on source-code corpora (code LLMs) can automate code generation, code explanation, and debugging tasks.

## Software Requirements
- Python 3.9 or above
- Hugging Face transformers library
- Pre-trained model: Salesforce/codegen-350M-mono (or OpenAI Codex/GPT API)
- Jupyter Notebook / VS Code

## Theory
Code-generation LLMs (e.g., CodeGen, StarCoder, Codex) are transformer decoder models pre-trained on large corpora of source code and natural-language comments. Given a natural-language instruction or a partial code snippet as prompt, the model predicts the most likely continuation, effectively 'writing' code.

For debugging, the buggy code along with an instruction (e.g., 'Fix the bug in this function') is passed as the prompt; the model uses patterns learned from millions of code examples to suggest corrected code or explain the likely cause of the error.

## Algorithm
1. Load a pre-trained code-generation model and its tokenizer.
2. Provide a natural-language instruction describing the required function.
3. Generate code using the model and display the output.
4. Provide a snippet of buggy code with an explicit 'fix the bug' instruction.
5. Generate the corrected code using the model.
6. Compare the buggy and fixed versions to verify correctness.

## How to Run

```bash
python code_assistant.py
```

## Sample Input
```
Instruction: 'check if a number is prime' + buggy factorial() function initialised with result = 0
```

## Sample Output
```
Generated Function:
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

Debug Suggestion:
def factorial_fixed(n):
    result = 1
    for i in range(1, n + 1):
        result = result * i
    return result
```

## Result
An AI-powered assistant was successfully developed that generates correct Python code from natural-language instructions and identifies/fixes bugs in existing code using a pre-trained code LLM.
