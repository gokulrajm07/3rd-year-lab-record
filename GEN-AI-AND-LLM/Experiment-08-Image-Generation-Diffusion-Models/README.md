[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KUMARAGURU-V-S/Lab-experiments/blob/main/GEN-AI-AND-LLM/Experiment-08-Image-Generation-Diffusion-Models/image_generation.ipynb)

# Experiment 08 - Image Generation Application Using Diffusion Models

## Aim
To implement an image generation application using a pre-trained Diffusion Model (Stable Diffusion) that synthesises images from text prompts.

## Objective
To understand the working principle of diffusion-based generative models — the forward noising process and the reverse denoising process — and to use them for text-to-image generation.

## Software Requirements
- Python 3.9 or above
- diffusers, transformers, torch, accelerate libraries
- Pre-trained model: runwayml/stable-diffusion-v1-5
- GPU-enabled environment (Google Colab with GPU runtime recommended)

## Theory
Diffusion models are generative models that learn to reverse a gradual noising process. During training, Gaussian noise is progressively added to real images over many timesteps (forward process) until the image becomes pure noise. The model — typically a U-Net — is trained to predict and remove this noise at each timestep (reverse process).

At inference time, the model starts from pure random noise and iteratively denoises it, step by step, guided by a text prompt encoded via a text encoder (e.g., CLIP), gradually 'sculpting' a coherent image that matches the prompt's description.

## Algorithm
1. Load the pre-trained Stable Diffusion pipeline onto the GPU.
2. Define a descriptive text prompt for the desired image.
3. Set the number of inference (denoising) steps and guidance scale.
4. Run the pipeline: start from random noise and iteratively denoise conditioned on the prompt.
5. Retrieve and save/display the final generated image.
6. Experiment with different prompts and guidance-scale values to observe quality changes.

## How to Run

```bash
python image_generation.py
```

## Sample Input
```
prompt = "A futuristic city skyline at sunset, digital art, highly detailed"
```

## Sample Output
```
Image generated and saved as generated_city.png
(A 512x512 PNG image showing a futuristic city skyline with warm sunset lighting is produced and saved to disk.)
```

## Result
An image generation application using the Stable Diffusion diffusion model was successfully implemented, generating a realistic image from a given text prompt.
