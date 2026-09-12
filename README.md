# TalStudio

Local Gradio app for generating images (Stable Diffusion v1.5) and short music
clips (MusicGen small) from text prompts.

Models are **lazy-loaded**: they only download/load into memory when you first
click Generate on that tab. This keeps startup fast and uses less RAM/VRAM until needed.

## Requirements

- Python 3.10+
- A CUDA GPU is strongly recommended (image/music generation will run on CPU
  but will be very slow). ~6–8GB VRAM is a reasonable minimum if you use one
  model at a time.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On GPU systems, install a CUDA build of PyTorch if needed:
https://pytorch.org/get-started/locally/

## Run

```bash
python app.py
```

Gradio will serve at `http://127.0.0.1:7860` by default.

## Notes

- Generated images and audio are saved under `image-generator/output/` and
  `music-generator/output/` with unique filenames (not committed).
- Both models use `float16` on CUDA to reduce VRAM usage.
- Stable Diffusion has attention slicing enabled for lower memory.
- The Stable Diffusion pipeline uses its default safety checker.
- First generation per modality will take longer while the model downloads/loads.

## License

MIT — see `LICENSE`.
