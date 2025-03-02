import modal, torch, io, ast, base64, pickle, re, numpy as np, nltk, os, time
from scipy.io import wavfile
from pydub import AudioSegment
# Rename the imported function to avoid the conflict
from bark import SAMPLE_RATE, preload_models, generate_audio as bark_generate_audio
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

def normalize_script_format(script_data):
    """Ensures the script is in the correct format of (speaker, text) tuples"""
    normalized_lines = []
    
    if not isinstance(script_data, list):
        print(f"Warning: Expected list but got {type(script_data)}")
        return [("Speaker 1", "Default audio content")]
    
    for item in script_data:
        if isinstance(item, tuple) and len(item) == 2:
            speaker, text = item
            
            # Handle special cases
            if isinstance(text, list) and len(text) > 0:
                # Extract text from list of dicts if needed
                if all(isinstance(dict_item, dict) for dict_item in text):
                    combined_text = ""
                    for dict_item in text:
                        if 'role' in dict_item and dict_item.get('role') == 'assistant':
                            content = dict_item.get('content', '')
                            if content and isinstance(content, str):
                                combined_text = content
                                break
                    
                    if combined_text:
                        normalized_lines.append((speaker, combined_text))
                    else:
                        normalized_lines.append((speaker, "Generated content"))
                else:
                    # Join list elements into a single string
                    normalized_lines.append((speaker, " ".join(str(x) for x in text)))
            else:
                # Use text as is if it's a string or convert to string otherwise
                normalized_lines.append((speaker, str(text)))
        else:
            print(f"Warning: Expected tuple but got {type(item)}")
    
    # Final validation
    if not normalized_lines:
        normalized_lines = [("Speaker 1", "Default audio content")]
    
    print(f"Normalized {len(normalized_lines)} lines of dialogue")
    return normalized_lines

@app.function(
    image=common_image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    container_idle_timeout=10*60,
    timeout=24*60*60,
    volumes={"/data": shared_volume},
    allow_concurrent_inputs=100,
)
def generate_audio(encoded_script: str, injection_id: str = None) -> str:
    """
    Takes the serialized script from generate_script() -> runs Bark TTS -> returns final .wav path
    """
    print(f"🔊 Bark TTS starting. Received encoded script of size: {len(encoded_script)} bytes")
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
                full_text = " ".join(str(item) for item in full_text)  # Join list into a single string

        voice_preset = speaker_voice_mapping.get(speaker, default_preset)
        sentences = sentence_splitter(preprocess_text(full_text))
        all_audio = []
        # For shorter silence (e.g., 0.1 seconds instead of 0.25)
        chunk_silence = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
        prev_generation_dict = None

        for sent in sentences:
            # Use bark_generate_audio instead of generate_audio
            generation_dict, audio_array = bark_generate_audio(
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

    # --- Step 1: Decode the serialized script ---
    try:
        print("Decoding base64 script data...")
        binary_data = base64.b64decode(encoded_script.encode('utf-8'))
        
        print("Unpickling script data...")
        script_data = pickle.loads(binary_data)
        
        print(f"Received script data type: {type(script_data)}")
        if isinstance(script_data, list):
            print(f"Script contains {len(script_data)} items")
            
        # Normalize the script format
        lines = normalize_script_format(script_data)
        print(f"Successfully decoded script with {len(lines)} dialogue lines")
        
    except Exception as e:
        print(f"❌ Error decoding serialized script: {e}")
        # Create a minimal fallback script
        lines = [
            ("Speaker 1", "Welcome to our podcast. Today we're discussing an interesting topic."),
            ("Speaker 2", "I'm excited to learn more about this. Could you tell us more?"),
            ("Speaker 1", "Of course! Let me explain some key points."),
            ("Speaker 2", "That's fascinating. What else should we know?")
        ]
        print("Using fallback script instead")

    # --- Step 2: Generate audio ---
    segments, rates = [], []
    for speaker, text in tqdm(lines, desc="🔊 Generating speech", unit="sentence"):
        arr, sr = generate_speaker_audio_longform(text, speaker)
        segments.append(arr)
        rates.append(sr)

    # --- Step 3: Merge into final audio file ---
    # Use injection_id in the filename if provided
    if injection_id:
        file_uuid = f"{injection_id}_{os.urandom(2).hex()}"
    else:
        file_uuid = os.urandom(4).hex()
        
    final_audio_path = os.path.join(AUDIO_OUTPUT_DIR, f"podcast_audio_{file_uuid}.wav")

    final_clip = concatenate_audio_segments(segments, rates)
    final_clip.export(final_audio_path, format="wav", codec="pcm_s16le")  # Ensure proper WAV encoding

    # Update database if injection_id is provided
    if injection_id:
        try:
            import sqlite3
            DB_PATH = "/data/injections.db"
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE injections SET processed_path = ?, status = 'completed' WHERE id = ?",
                (final_audio_path, injection_id)
            )
            conn.commit()
            conn.close()
            print(f"✅ Database updated for injection ID: {injection_id}")
        except Exception as e:
            print(f"⚠️ Error updating database: {e}")

    # Explicitly commit volume changes so other containers can access it
    shared_volume.commit()

    print(f"✅ Done. Final audio saved at: {final_audio_path}")
    return final_audio_path
