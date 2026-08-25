[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KUMARAGURU-V-S/Lab-experiments/blob/main/GEN-AI-AND-LLM/Experiment-09-Multimodal-AI-Application/multimodal_app.ipynb)

# Experiment 09 - Multimodal AI Application Integrating Text and Image Inputs

## Aim
To develop a multimodal AI application that processes both text and image inputs using a pre-trained vision-language model (BLIP) to perform image captioning and visual question answering.

## Objective
To understand how multimodal foundation models jointly encode visual and textual information into a shared representation space to perform cross-modal tasks.

## Software Requirements
- Python 3.9 or above
- Hugging Face transformers, Pillow libraries
- Pre-trained model: Salesforce/blip-image-captioning-base, Salesforce/blip-vqa-base
- Jupyter Notebook / VS Code

## Theory
Multimodal AI systems combine information from more than one modality — such as text and image — to perform tasks that neither modality could solve alone. Vision-language models like BLIP (Bootstrapped Language-Image Pre-training) use a vision encoder (typically a Vision Transformer) to convert an image into a sequence of visual features, and a text encoder/decoder to process language.

These visual and textual representations are fused (often via cross-attention layers) into a joint embedding space, enabling tasks such as image captioning (generating a text description of an image) and Visual Question Answering — VQA (answering a natural-language question about the content of an image).

## Algorithm
1. Load a pre-trained BLIP model and processor for image captioning.
2. Load an input image using PIL.
3. Pass the image through the processor and model to generate a caption.
4. Load a BLIP-VQA model and processor.
5. Provide the same image along with a natural-language question.
6. Generate and display the answer produced by the model.

## How to Run

```bash
python multimodal_app.py
```

## Sample Input
```
Image of a dog in a field (URL) + Question: 'What animal is in the picture?'
```

## Sample Output
```
Generated Caption: a dog running through a grassy field
Question: What animal is in the picture?
Answer: dog
```

## Result
A multimodal AI application integrating text and image inputs was successfully developed using the BLIP model, performing both image captioning and visual question answering.
