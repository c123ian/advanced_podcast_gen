import modal, torch, io, ast, base64, pickle, re, numpy as np, nltk, os, time
from scipy.io import wavfile
from pydub import AudioSegment
# Import the specific low-level Bark functions
from bark import SAMPLE_RATE, preload_models
from bark.generation import generate_text_semantic
from bark.api import semantic_to_waveform
from tqdm import tqdm
from typing import List, Tuple

# Import shared resources
from common_image import common_image, shared_volume

app = modal.App("audio_gen")
# Create a shared dictionary for voice states (RNG seeds)
voice_states = modal.Dict.from_name("voice-states", create_if_missing=True)

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


def estimate_processing_time(lines):
    """Provide a rough ETA based on script length"""
    seconds_per_line = 50  # Your estimated processing time per line
    total_seconds = len(lines) * seconds_per_line
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes} minutes, {seconds} seconds"


def convert_disfluencies(text):
    """
    Convert parenthesized expressions like (laughs) to bracketed [laughs]
    for proper TTS rendering.
    """
    # List of common disfluencies to check for
    disfluencies = [
        "laughs", "sighs", "laughter", "gasps", "clears throat", 
        "sigh", "laugh", "gasp", "chuckles", "snorts",
        "hmm", "umm", "uh", "ah", "er", "um"
    ]
    
    # Convert (laughs) to [laughs]
    for disfluency in disfluencies:
        # Look for various formats and convert them
        text = re.sub(r'\((' + disfluency + r')\)', r'[\1]', text, flags=re.IGNORECASE)
        text = re.sub(r'<(' + disfluency + r')>', r'[\1]', text, flags=re.IGNORECASE)
        
        # Also match when there's text inside
        text = re.sub(r'\(([^)]*' + disfluency + r'[^)]*)\)', r'[\1]', text, flags=re.IGNORECASE)
        
    return text


# Function to process a specific speaker's lines in parallel
@app.function(
    image=common_image,
    gpu=modal.gpu.A10G(count=1),
    timeout=24*60*60,
    volumes={"/data": shared_volume},
)
def generate_speaker_audio(speaker_lines: List[Tuple[int, str]], 
                          speaker: str, 
                          injection_id: str = None) -> List[Tuple[int, np.ndarray, int]]:
    """Generate audio for all lines from a single speaker"""
    print(f"🎙️ Starting audio generation for {speaker} with {len(speaker_lines)} lines")
    preload_models()  # Ensure Bark model is loaded

    def sentence_splitter(text):
        return nltk.sent_tokenize(text)

    def preprocess_text(text):
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return re.sub(r"[^\w\s.,!?-]", "", text)

    # Speaker voice presets
    speaker_voice_mapping = {"Speaker 1": "v2/en_speaker_9", "Speaker 2": "v2/en_speaker_6"}
    default_preset = "v2/en_speaker_9"
    
    # Get voice preset for this speaker
    voice_preset = speaker_voice_mapping.get(speaker, default_preset)
    
    # Setup voice consistency with RNG seed
    voice_state_key = f"{injection_id}_{speaker}" if injection_id else None
    
    if voice_state_key and voice_states.contains(voice_state_key):
        seed = voice_states.get(voice_state_key)
        print(f"Using saved seed {seed} for {speaker}")
    else:
        speaker_num = 1 if speaker == "Speaker 1" else 2
        seed = np.random.randint(10000 * speaker_num, 10000 * (speaker_num + 1) - 1)
        if voice_state_key:
            voice_states[voice_state_key] = seed
        print(f"Created new seed {seed} for {speaker}")
    
    # Set seed for consistent voice
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generation parameters
    text_temp = 0.6
    waveform_temp = 0.6
    
    # Process each line for this speaker
    results = []
    for line_idx, text in tqdm(speaker_lines, desc=f"Generating {speaker} audio"):
        # Process disfluencies in the text
        text = convert_disfluencies(text)
        sentences = sentence_splitter(preprocess_text(text))
        all_audio = []
        chunk_silence = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
        
        for i, sent in enumerate(sentences):
            semantic_tokens = generate_text_semantic(
                sent,
                history_prompt=voice_preset,
                temp=text_temp,
                min_eos_p=0.05,
            )
            
            audio_array = semantic_to_waveform(
                semantic_tokens, 
                history_prompt=voice_preset,
                temp=waveform_temp,
            )
            
            all_audio.append(audio_array)
            if i < len(sentences) - 1:  # Don't add silence after the last sentence
                all_audio.append(chunk_silence)
        
        if not all_audio:
            line_audio = np.zeros(24000, dtype=np.float32)
        else:
            line_audio = np.concatenate(all_audio, axis=0)
        
        # Store with original index for reassembly
        results.append((line_idx, line_audio, SAMPLE_RATE))
        print(f"Completed line {line_idx} for {speaker}")
    
    print(f"✅ Completed all {len(speaker_lines)} lines for {speaker}")
    return results


@app.function(
    image=common_image,
    # No GPU needed for the coordinator
    container_idle_timeout=10*60,
    timeout=24*60*60,
    volumes={"/data": shared_volume},
    allow_concurrent_inputs=100,
)
def generate_audio(encoded_script: str, injection_id: str = None) -> str:
    """
    Takes the serialized script from generate_script() -> runs Bark TTS -> returns final .wav path
    """    
    print(f"🔊 Bark TTS starting with parallel processing. Received encoded script of size: {len(encoded_script)} bytes")

    def numpy_to_audio_segment(audio_arr, sr):
        """Converts numpy audio array to an AudioSegment"""
        audio_int16 = (audio_arr * 32767).astype(np.int16)
        bio = io.BytesIO()
        wavfile.write(bio, sr, audio_int16)
        bio.seek(0)
        return AudioSegment.from_wav(bio)

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

        # 50 seconds per line TTS generation estimate
        estimated_time = estimate_processing_time(lines)
        print(f"✨ Estimated processing time: {estimated_time} for {len(lines)} lines")
        
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

    # --- SPEAKER SEPARATION FOR PARALLEL PROCESSING ---
    # Original script format is a list of tuples like:
    # [
    #   ("Speaker 1", "Welcome to our podcast..."),  # index 0
    #   ("Speaker 2", "Thanks for having me..."),    # index 1
    #   ("Speaker 1", "Today we're discussing..."),  # index 2
    #   ...
    # ]

    # Print the first few lines to verify input format
    print("DEBUG - Original script format:")
    for i, (speaker, text) in enumerate(lines[:min(3, len(lines))]):
        print(f"  Line {i}: {speaker} says: {text[:30]}...")

    # Split by speaker, keeping original indices for later reassembly
    speaker1_lines = [(i, text) for i, (speaker, text) in enumerate(lines) if speaker == "Speaker 1"]
    speaker2_lines = [(i, text) for i, (speaker, text) in enumerate(lines) if speaker == "Speaker 2"]

    # Print the split results to verify
    print(f"\nDEBUG - Speaker 1 lines ({len(speaker1_lines)} total):")
    for i, text in speaker1_lines[:min(3, len(speaker1_lines))]:
        print(f"  Original index {i}: {text[:30]}...")

    print(f"\nDEBUG - Speaker 2 lines ({len(speaker2_lines)} total):")
    for i, text in speaker2_lines[:min(3, len(speaker2_lines))]:
        print(f"  Original index {i}: {text[:30]}...")

    # === PARALLEL GPU PROCESSING EXPLANATION ===
    # This is how we process Speaker 1 and Speaker 2 lines simultaneously on separate GPUs:
    #
    # 1. Function Definition:
    #    @app.function(gpu=modal.gpu.A100(count=1))
    #    def generate_speaker_audio(...):
    #
    #    The "count=1" means each CONTAINER gets 1 GPU, not that the entire app uses only 1 GPU.
    #
    # 2. Parallel Execution:
    #    speaker1_results = generate_speaker_audio.remote(speaker1_lines, "Speaker 1", injection_id)
    #    speaker2_results = generate_speaker_audio.remote(speaker2_lines, "Speaker 2", injection_id)
    #
    #    Each .remote() call spawns a separate container with its own dedicated GPU.
    #    These containers run simultaneously in Modal's cloud infrastructure.
    #
    # 3. Result:
    #    - Speaker 1 lines process on GPU #1 in Container #1
    #    - Speaker 2 lines process on GPU #2 in Container #2
    #    - Both run in parallel, potentially cutting processing time nearly in half
    #
    # This is much more efficient than processing all lines sequentially on a single GPU.

    print("\nDEBUG - Starting parallel processing of speakers:")
    # Process each speaker's lines in parallel
    speaker1_results = []
    speaker2_results = []

    # Only process if there are lines for that speaker
    if speaker1_lines:
        print(f"  Sending {len(speaker1_lines)} Speaker 1 lines to GPU #1")
        speaker1_results = generate_speaker_audio.remote(speaker1_lines, "Speaker 1", injection_id)

    if speaker2_lines:
        print(f"  Sending {len(speaker2_lines)} Speaker 2 lines to GPU #2")
        speaker2_results = generate_speaker_audio.remote(speaker2_lines, "Speaker 2", injection_id)

    # --- Step 4: Combine results in original script order ---
    all_results = speaker1_results + speaker2_results
    all_results.sort(key=lambda x: x[0])  # Sort by original script line index

    print(f"\nDEBUG - Received {len(all_results)} total audio segments from parallel processing")
    print("  Recombining in original script order based on line indices...")

    # --- Step 5: Assemble final audio ---
    segments = []
    rates = []
    turn_silence = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)  # Silence between turns
    
    for _, audio_array, sr in all_results:
        segments.append(audio_array)
        segments.append(turn_silence)  # Add half-second silence between turns
        rates.append(sr)
        rates.append(sr)  # Need matching rate for silence

    # --- Step 6: Save to file ---
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

    # At the end of the function, make sure the database is updated:
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
            print(f"✅ Database updated for injection ID: {injection_id} - Status: completed")
        except Exception as e:
            print(f"⚠️ Error updating database: {e}")
    
    # Clean up voice state data after successful completion
    if injection_id:
        # Remove all voice states for this injection to free up space
        speaker_voice_mapping = {"Speaker 1": "v2/en_speaker_9", "Speaker 2": "v2/en_speaker_6"}
        for speaker in speaker_voice_mapping.keys():
            voice_state_key = f"{injection_id}_{speaker}"
            if voice_states.contains(voice_state_key):
                voice_states.pop(voice_state_key)
                print(f"Cleaned up voice state for {voice_state_key}")

    print(f"✅ Done. Final audio saved at: {final_audio_path}")
    return final_audio_path
