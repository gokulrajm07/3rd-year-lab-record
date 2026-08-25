[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KUMARAGURU-V-S/Lab-experiments/blob/main/GEN-AI-AND-LLM/Experiment-03-Conversational-AI-Chatbot/chatbot.ipynb)

# Experiment 03 - Conversational AI Chatbot Using Transformer-Based Language Models

## Aim
To build a conversational AI chatbot capable of holding a multi-turn dialogue using a transformer-based language model (DialoGPT).

## Objective
To understand how transformer decoder models maintain conversational context across multiple turns using dialogue history encoding, and to implement a simple interactive chatbot.

## Software Requirements
- Python 3.9 or above
- Hugging Face transformers and torch libraries
- Pre-trained model: microsoft/DialoGPT-medium
- Jupyter Notebook / VS Code

## Theory
Conversational AI chatbots use transformer-based language models trained on large volumes of dialogue data to generate human-like responses. DialoGPT is a GPT-2-based model fine-tuned on Reddit conversation threads, specifically designed for open-domain, multi-turn dialogue generation.

To maintain context, the chatbot concatenates the current user input with the previous conversation turns (encoded as token ids) before generating a response, so that the model 'remembers' what was said earlier in the session. The response is generated using sampling or beam search and decoded back into readable text.

## Algorithm
1. Load the DialoGPT tokenizer and model.
2. Initialise an empty chat-history tensor.
3. In a loop, accept user input and encode it, appending an end-of-sentence token.
4. Concatenate the new input with the existing chat history.
5. Generate a response from the model using the combined history as context.
6. Decode and display only the newly generated tokens as the bot's reply.
7. Update the chat history and repeat for the next turn.

## How to Run

```bash
python chatbot.py
```

## Sample Input
```
>> User: Hi, how are you?
>> User: What can you help me with?
```

## Sample Output
```
Bot: I'm doing great, thanks for asking! How about you?
Bot: I can chat with you about almost anything - just ask away!
```

## Result
A multi-turn conversational chatbot was successfully built using the transformer-based DialoGPT model, capable of retaining dialogue context across turns.
