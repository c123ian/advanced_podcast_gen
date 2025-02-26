import modal, torch, io, ast, base64, pickle, re, numpy as np, nltk, os
from scipy.io import wavfile
from pydub import AudioSegment
from bark import SAMPLE_RATE, preload_models, generate_audio
from tqdm import tqdm

# Import shared resources
from common_image import common_image, shared_volume

app = modal.App("audio_gen")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup NLTK for sentence splitting
NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)
nltk.download("punkt", download_dir=NLTK_DATA_DIR)
nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)

# Define persistent storage path for audio files
AUDIO_OUTPUT_DIR = "/data/podcast_audio"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

@app.function(
    image=common_image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    container_idle_timeout=10*60,
    timeout=24*60*60,
    volumes={"/data": shared_volume},
    allow_concurrent_inputs=100,
)
def generate_audio(script_pkl_path: str) -> str:
    """
    Takes the .pkl from generate_script() -> runs Bark TTS -> returns final .wav path
    """
    print(f"🔊 Bark TTS starting. script_pkl_path={script_pkl_path}")
    preload_models()  # Ensure Bark model is loaded

    def sentence_splitter(text):
        return nltk.sent_tokenize(text)

    def preprocess_text(text):
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return re.sub(r"[^\w\s.,!?-]", "", text)

    def numpy_to_audio_segment(audio_arr, sr):
        """Converts numpy audio array to an AudioSegment"""
        audio_int16 = (audio_arr * 32767).astype(np.int16)
        bio = io.BytesIO()
        wavfile.write(bio, sr, audio_int16)
        bio.seek(0)
        return AudioSegment.from_wav(bio)

    # Speaker voice presets
    speaker_voice_mapping = {"Speaker 1": "v2/en_speaker_9", "Speaker 2": "v2/en_speaker_6"}
    default_preset = "v2/en_speaker_9"

    def generate_speaker_audio_longform(full_text, speaker):
        """Generates TTS audio for a given speaker and text"""
        if isinstance(full_text, list):
            # Check if the list contains dictionaries and extract the text
            if all(isinstance(item, dict) for item in full_text):
                full_text = " ".join(item.get('text', '') for item in full_text)  # Join dicts into a single string
            else:
                full_text = " ".join(full_text)  # Join list into a single string

        voice_preset = speaker_voice_mapping.get(speaker, default_preset)
        sentences = sentence_splitter(preprocess_text(full_text))
        all_audio = []
        chunk_silence = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)
        prev_generation_dict = None

        for sent in sentences:
            generation_dict, audio_array = generate_audio(
                text=sent,
                history_prompt=prev_generation_dict if prev_generation_dict else voice_preset,
                output_full=True,
                text_temp=0.7,
                waveform_temp=0.7,
            )
            prev_generation_dict = generation_dict
            all_audio.append(audio_array)
            all_audio.append(chunk_silence)

        if not all_audio:
            return np.zeros(24000, dtype=np.float32), SAMPLE_RATE

        return np.concatenate(all_audio, axis=0), SAMPLE_RATE

    def concatenate_audio_segments(segments, rates):
        """Concatenates multiple audio segments"""
        final_audio = None
        for seg, sr in zip(segments, rates):
            seg_audio = numpy_to_audio_segment(seg, sr)
            final_audio = seg_audio if final_audio is None else final_audio.append(seg_audio, crossfade=100)
        return final_audio

    # --- Step 1: Load script ---
    try:
        with open(script_pkl_path, "rb") as f:
            lines = pickle.load(f)  # Should be a list of (speaker, text) pairs
        if not isinstance(lines, list):
            raise ValueError(f"Loaded script is not a list: {type(lines)}")
    except Exception as e:
        raise RuntimeError(f"Error loading .pkl script: {e}")

    # --- Step 2: Generate audio ---
    segments, rates = [], []
    for speaker, text in tqdm(lines, desc="🔊 Generating speech", unit="sentence"):
        arr, sr = generate_speaker_audio_longform(text, speaker)
        segments.append(arr)
        rates.append(sr)

    # --- Step 3: Merge into final audio file ---
    file_uuid = "finalwav_" + os.urandom(4).hex()
    final_audio_path = os.path.join(AUDIO_OUTPUT_DIR, f"final_podcast_audio_{file_uuid}.wav")

    final_clip = concatenate_audio_segments(segments, rates)
    final_clip.export(final_audio_path, format="wav", codec="pcm_s16le")  # Ensure proper WAV encoding

    # Explicitly commit volume changes so other containers can access it
    shared_volume.commit()

    print(f"✅ Done. Final audio saved at: {final_audio_path}")
    return final_audio_path

