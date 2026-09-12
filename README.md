# TalStudio

Local Gradio app for generating images (Stable Diffusion v1.5) and short music clips (MusicGen small) from text prompts.

Models are **lazy-loaded**: they only download/load into memory when you first click **Generate** on that tab. This keeps startup fast and uses less RAM/VRAM until needed.

## Requirements

- Python 3.10 – 3.14
- NVIDIA GPU with CUDA strongly recommended (RTX 4060 8GB works well)
- ~6–8 GB VRAM is enough if you generate one modality at a time

> CPU mode works but is **very slow**.

## Setup (Windows recommended)

```powershell
# 1. Create virtual environment (recommended)
python -m venv .venv
.\v.venv\Scripts\Activate.ps1

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install CUDA PyTorch (important!)
# For Python 3.14 use cu130 (or cu128 / cu132)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 4. Install the rest
pip install -r requirements.txt
```

Check that GPU is detected:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU yok')"
```

You should see `CUDA: True` and your GPU name.

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130   # or appropriate CUDA version
pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Open in browser: [http://127.0.0.1:7860](http://127.0.0.1:7860)

## Tips for better images

- Write detailed prompts (subject + style + lighting + quality words)
- Always fill the **Negative Prompt** (blurry, low quality, deformed, watermark…)
- Steps: 25–35  |  Guidance: 7.5–9  works well for SD 1.5

Example good prompt:
```
a flying mallard duck, wings spread, soaring in blue sky with clouds, detailed feathers, realistic, cinematic lighting
```

Negative:
```
blurry, low quality, deformed, ugly, watermark, text, sea, ocean, water
```

## Notes

- Generated files are saved in:
  - `image-generator/output/`
  - `music-generator/output/`
- Both models use `float16` on CUDA + attention slicing to save VRAM
- First generation of each type is slower (model download + load)
- If you get **OSError 1455** (page file too small) on Windows → increase Virtual Memory (page file) and restart

## License

MIT — see `LICENSE`.
