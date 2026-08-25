"""
Experiment 01: Text Generation Using Pre-Trained Foundation Models
Course: CS4V48 - GenAI & LLM Laboratory
"""

from transformers import pipeline, set_seed

def main():
    # Load the pre-trained GPT-2 text generation pipeline
    print("Loading GPT-2 model pipeline...")
    generator = pipeline("text-generation", model="gpt2")
    set_seed(42)

    # Input prompt
    prompt = "Artificial Intelligence will transform the future of"
    print(f"Prompt: '{prompt}'\n")

    # Generate text with decoding strategies (sampling, top_k, top_p, temperature)
    outputs = generator(
        prompt,
        max_length=60,
        num_return_sequences=2,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )

    for i, out in enumerate(outputs, 1):
        print(f"--- Generated Text {i} ---")
        print(out["generated_text"])
        print()

if __name__ == "__main__":
    main()
