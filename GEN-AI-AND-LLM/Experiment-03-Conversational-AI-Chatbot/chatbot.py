"""
Experiment 03: Conversational AI Chatbot Using Transformer-Based Language Models
Course: CS4V48 - GenAI & LLM Laboratory
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("Loading DialoGPT-medium model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    chat_history_ids = None
    print("Chatbot ready! Type 'quit' or 'exit' to stop.\n")

    for step in range(5):
        user_input = input(">> User: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

        bot_input_ids = (
            torch.cat([chat_history_ids, new_input_ids], dim=-1)
            if chat_history_ids is not None
            else new_input_ids
        )

        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.9
        )

        response = tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()
