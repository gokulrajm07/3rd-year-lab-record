"""
Experiment 07: AI-Powered Code Generation and Debugging Assistant
Course: CS4V48 - GenAI & LLM Laboratory
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_code(model, tokenizer, prompt, max_new_tokens=80):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    print("Loading Salesforce/codegen-350M-mono model...")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

    # 1. Code generation from natural language instruction
    print("\n--- Task 1: Code Generation ---")
    prompt1 = "# Write a Python function to check if a number is prime\ndef is_prime(n):"
    print("Generated Function:")
    print(generate_code(model, tokenizer, prompt1))

    # 2. Debugging buggy snippet
    print("\n--- Task 2: Code Debugging ---")
    buggy_code = (
        "# The following function should return the factorial of n, but has a bug. Fix it.\n"
        "def factorial(n):\n"
        "    result = 0\n"
        "    for i in range(1, n + 1):\n"
        "        result = result * i\n"
        "    return result\n\n"
        "# Corrected function:\n"
        "def factorial_fixed(n):"
    )
    print("Debug Suggestion:")
    print(generate_code(model, tokenizer, buggy_code, max_new_tokens=60))

if __name__ == "__main__":
    main()
