[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KUMARAGURU-V-S/Lab-experiments/blob/main/GEN-AI-AND-LLM/Experiment-11-Content-Generation-Multimedia/multimedia_gen.ipynb)

# Experiment 11 - AI-Based Content Generation System for Text, Image and Multimedia Applications

## Aim
To develop an integrated AI-based content generation system that produces text, image and audio content from a single high-level user input.

## Objective
To understand how multiple generative models — an LLM for text, a diffusion model for images, and a text-to-speech model for audio — can be orchestrated together to build an end-to-end multimedia content-generation pipeline.

## Software Requirements
- Python 3.9 or above
- transformers, diffusers, gTTS (or a TTS model) libraries
- Pre-trained models: gpt2/flan-t5 (text), stable-diffusion-v1-5 (image), gTTS (audio)
- Jupyter Notebook / VS Code, GPU recommended for image generation

## Theory
AI-based content generation systems automate the creation of blog posts, social-media captions, marketing images, and narrated audio by chaining together multiple generative models, each specialised for one modality. A pipeline approach is typically used: the LLM first generates textual content (e.g., a short article or caption) from a topic; that generated text (or a derived prompt) is then passed to an image-generation model to create matching visuals; finally, a text-to-speech (TTS) model converts the generated text into narrated audio.

## Algorithm
1. Accept a high-level topic/theme from the user.
2. Use a text-generation LLM to produce a short article/caption on the topic.
3. Derive an image prompt from the generated text.
4. Use a diffusion model to generate an accompanying image from the image prompt.
5. Convert the generated text into speech using a text-to-speech model.
6. Save/display the generated text, image, and audio as the final multimedia content package.

## How to Run

```bash
python multimedia_gen.py
```

## Sample Input
```
topic = "The benefits of renewable energy"
```

## Sample Output
```
Generated Text:
Renewable energy sources like solar and wind reduce carbon emissions, lower energy costs over time, and help create a sustainable future for generations to come.

Image saved as content_image.png
Audio saved as content_audio.mp3
```

## Result
An integrated AI-based content generation system producing text, image, and audio outputs from a single topic input was successfully developed.
