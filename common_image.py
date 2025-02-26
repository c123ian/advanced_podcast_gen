# my_podcast_proj/common_image.py

import modal

try:
    shared_volume = modal.Volume.lookup("combined_volume")
except modal.exception.NotFoundError:
    shared_volume = modal.Volume.persisted("combined_volume")

common_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")  
    .pip_install(
        "torch==2.5.1",  # temp fic to avoid stricter security check by torch: https://github.com/suno-ai/bark/pull/619
        "PyPDF2",
        "python-fasthtml==0.12.0",
        "langchain",
        "langchain-community",
        "openai-whisper",
        "beautifulsoup4",
        "requests",
        "pydub",
        "nltk",
        "tqdm",
        "scipy",
        "transformers==4.46.1",
        "accelerate==1.2.1",
        "git+https://github.com/suno-ai/bark.git"
    )
)

