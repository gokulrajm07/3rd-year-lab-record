"""
Experiment 02: Prompt Engineering Techniques for Content Generation, Reasoning and Task Automation
Course: CS4V48 - GenAI & LLM Laboratory
"""

from transformers import pipeline

def main():
    print("Loading GPT-2 model pipeline...")
    generator = pipeline("text-generation", model="gpt2")

    # 1. Zero-shot prompt
    zero_shot_prompt = (
        "Classify the sentiment of this review as Positive or Negative:\n"
        "'The product quality is excellent!'\n"
        "Sentiment:"
    )

    # 2. Few-shot prompt
    few_shot_prompt = (
        "Review: 'I loved this movie, it was fantastic.'\n"
        "Sentiment: Positive\n\n"
        "Review: 'The service was slow and disappointing.'\n"
        "Sentiment: Negative\n\n"
        "Review: 'The product quality is excellent!'\n"
        "Sentiment:"
    )

    # 3. Chain-of-Thought prompt
    cot_prompt = (
        "Q: A shop had 15 apples. It sold 6 and then received 10 more. How many apples now?\n"
        "A: Let's think step by step. 15 - 6 = 9. 9 + 10 = 19. The answer is 19.\n\n"
        "Q: A library had 120 books. It lent out 45 and bought 30 new books. How many books now?\n"
        "A: Let's think step by step."
    )

    prompts = [
        ("Zero-shot", zero_shot_prompt),
        ("Few-shot", few_shot_prompt),
        ("Chain-of-Thought", cot_prompt),
    ]

    for name, p in prompts:
        out = generator(p, max_length=len(p.split()) + 40, num_return_sequences=1, do_sample=False)
        print(f"=== {name} ===")
        print(out[0]["generated_text"])
        print()

if __name__ == "__main__":
    main()
