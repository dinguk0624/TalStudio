import os
import random
import uuid

import gradio as gr
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGE_OUTPUT = os.path.join(ROOT, "image-generator", "output")
MUSIC_OUTPUT = os.path.join(ROOT, "music-generator", "output")

os.makedirs(IMAGE_OUTPUT, exist_ok=True)
os.makedirs(MUSIC_OUTPUT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

IMAGE_MODEL = "runwayml/stable-diffusion-v1-5"
MUSIC_MODEL = "facebook/musicgen-small"

# Lazy-loaded model handles (None until first use)
_image_pipe = None
_music_processor = None
_music_model = None

print(f"Using device: {DEVICE} (dtype: {DTYPE})")
print("Models will load on first generate (lazy load).")


def _get_image_pipe():
    """Load Stable Diffusion only when image generation is requested."""
    global _image_pipe
    if _image_pipe is not None:
        return _image_pipe

    from diffusers import StableDiffusionPipeline

    print("Loading image model (Stable Diffusion v1.5)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        IMAGE_MODEL,
        torch_dtype=DTYPE,
    )
    pipe = pipe.to(DEVICE)

    # Lower VRAM usage
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    _image_pipe = pipe
    print("Image model loaded.")
    return _image_pipe


def _get_music_model():
    """Load MusicGen only when music generation is requested."""
    global _music_processor, _music_model
    if _music_model is not None:
        return _music_processor, _music_model

    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    print("Loading music model (MusicGen small)...")
    processor = AutoProcessor.from_pretrained(MUSIC_MODEL)
    model = MusicgenForConditionalGeneration.from_pretrained(
        MUSIC_MODEL,
        torch_dtype=DTYPE,
    )
    model = model.to(DEVICE)
    _music_processor = processor
    _music_model = model
    print("Music model loaded.")
    return _music_processor, _music_model


def generate_image(prompt, negative_prompt, steps, guidance, seed):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter an image prompt.")

    if seed == -1:
        seed = random.randint(0, 2**31 - 1)
    seed = int(seed)

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    try:
        pipe = _get_image_pipe()
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=generator,
        )
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error(
            "Out of GPU memory. Try lowering steps, using a shorter prompt, "
            "or restarting the app."
        )
    except Exception as e:
        raise gr.Error(f"Image generation failed: {str(e)}")

    image = result.images[0]

    nsfw = getattr(result, "nsfw_content_detected", None)
    if nsfw and nsfw[0]:
        raise gr.Error(
            "The generated image was flagged by the safety checker. "
            "Try a different prompt."
        )

    output_path = os.path.join(
        IMAGE_OUTPUT, f"talstudio_image_{uuid.uuid4().hex}.png"
    )
    image.save(output_path)
    return image, output_path, seed


def generate_music(prompt, duration, seed):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a music prompt.")

    if seed == -1:
        seed = random.randint(0, 2**31 - 1)
    seed = int(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        processor, model = _get_music_model()
        inputs = processor(text=[prompt], padding=True, return_tensors="pt")
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        # MusicGen ≈ 50 tokens per second of audio
        max_new_tokens = max(50, int(duration) * 50)

        with torch.inference_mode():
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=3.0,
                max_new_tokens=max_new_tokens,
            )
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error(
            "Out of GPU memory. Try a shorter duration or restarting the app."
        )
    except Exception as e:
        raise gr.Error(f"Music generation failed: {str(e)}")

    import soundfile as sf

    # audio_values shape: (batch, channels, samples) or (batch, samples)
    audio = audio_values[0].detach().cpu().float().numpy()

    if audio.ndim == 2:
        # (channels, samples) -> (samples, channels) for soundfile
        if audio.shape[0] <= 2:  # stereo/mono channels first
            audio = audio.T

    sample_rate = model.config.audio_encoder.sampling_rate
    output_path = os.path.join(
        MUSIC_OUTPUT, f"talstudio_music_{uuid.uuid4().hex}.wav"
    )
    sf.write(output_path, audio, sample_rate)
    return output_path, output_path, seed


css = """
.gradio-container {
    background: linear-gradient(135deg, #0f172a, #111827);
    color: white;
    font-family: Inter, Arial, sans-serif;
    max-width: 1200px !important;
}

#title {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    color: #f9a8d4;
}

#subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 18px;
}

.generate-button {
    background: linear-gradient(135deg, #ec4899, #8b5cf6) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
}
"""


with gr.Blocks(
    css=css,
    theme=gr.themes.Soft(),
    title="TalStudio",
) as demo:

    gr.Markdown("# TalStudio", elem_id="title")
    gr.Markdown(
        "A local AI studio for generating images and music.\n"
        "Models load only when you first click Generate (saves RAM/VRAM).",
        elem_id="subtitle",
    )

    with gr.Tabs():
        with gr.Tab("Image Generator"):
            gr.Markdown("## Image Generator")
            gr.Markdown(
                "Create images from text prompts using a local Stable Diffusion model. "
                "The model loads on first use."
            )

            with gr.Row():
                image_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder=(
                        "Example: a futuristic city at night, "
                        "neon lights, cinematic lighting"
                    ),
                    lines=4,
                )
                image_negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder=(
                        "Example: blurry, low quality, "
                        "watermark, text, logo"
                    ),
                    lines=4,
                )

            with gr.Row():
                image_steps = gr.Slider(
                    minimum=10, maximum=50, value=25, step=1, label="Steps"
                )
                image_guidance = gr.Slider(
                    minimum=1, maximum=20, value=7.5, step=0.5, label="Guidance Scale"
                )
                image_seed = gr.Number(
                    value=-1, precision=0, label="Seed (-1 = random)"
                )

            image_button = gr.Button(
                "Generate Image",
                variant="primary",
                elem_classes="generate-button",
            )
            image_output = gr.Image(label="Generated Image", type="pil")
            image_file = gr.File(label="Saved Image")
            image_used_seed = gr.Number(label="Used Seed", interactive=False)

            image_button.click(
                fn=generate_image,
                inputs=[
                    image_prompt,
                    image_negative_prompt,
                    image_steps,
                    image_guidance,
                    image_seed,
                ],
                outputs=[image_output, image_file, image_used_seed],
            )

        with gr.Tab("Music Generator"):
            gr.Markdown("## Music Generator")
            gr.Markdown(
                "Create short music clips from text prompts using MusicGen. "
                "The model loads on first use."
            )

            music_prompt = gr.Textbox(
                label="Prompt",
                placeholder=(
                    "Example: calm lo-fi instrumental with soft piano, "
                    "warm bass and gentle drums"
                ),
                lines=4,
            )

            with gr.Row():
                music_duration = gr.Slider(
                    minimum=4, maximum=16, value=8, step=1, label="Duration (seconds)"
                )
                music_seed = gr.Number(
                    value=-1, precision=0, label="Seed (-1 = random)"
                )

            music_button = gr.Button(
                "Generate Music",
                variant="primary",
                elem_classes="generate-button",
            )
            music_output = gr.Audio(label="Generated Music", type="filepath")
            music_file = gr.File(label="Saved Audio File")
            music_used_seed = gr.Number(label="Used Seed", interactive=False)

            music_button.click(
                fn=generate_music,
                inputs=[music_prompt, music_duration, music_seed],
                outputs=[music_output, music_file, music_used_seed],
            )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
