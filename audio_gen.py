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

# Try to look up the Bark models volume
try:
    bark_volume = modal.Volume.lookup("bark_models")
    print("Found existing bark_models volume")
except modal.exception.NotFoundError:
    print("Warning: bark_models volume not found. Please run download_bark_models.py first")
    bark_volume = None

app = modal.App("audio_gen")
# Create a shared dictionary for voice states (RNG seeds)
voice_states = modal.Dict.from_name("voice-states", create_if_missing=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Script length limits for safety
MAX_SCRIPT_LINES = 30  # Absolute maximum for safety
TARGET_SCRIPT_LINES = 15  # Ideal length for a podcast

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
    volumes={
        "/data": shared_volume,
        "/bark_models": bark_volume  # Add the Bark models volume
    } if bark_volume else {"/data": shared_volume},
)
def generate_speaker_audio(speaker_lines: List[Tuple[int, str]], 
                          speaker: str, 
                          injection_id: str = None) -> List[Tuple[int, np.ndarray, int]]:
    """Generate audio for all lines from a single speaker with progress updates"""
    print(f"🎙️ Starting audio generation for {speaker} with {len(speaker_lines)} lines")
    
    # Set environment variables to use the pre-downloaded models if available
    if os.path.exists('/bark_models'):
        print("Using pre-downloaded Bark models from volume")
        os.environ["XDG_CACHE_HOME"] = "/bark_models"
    
    preload_models()  # Ensure Bark model is loaded

    def update_speaker_status(progress_msg):
        """Helper to update database with current speaker progress"""
        if not injection_id:
            return
            
        try:
            import sqlite3
            DB_PATH = "/data/injections_truncate.db"
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # First get the current status and notes
            cursor.execute(
                "SELECT status, processing_notes FROM injections WHERE id = ?",
                (injection_id,)
            )
            result = cursor.fetchone()
            
            if result:
                status, notes = result
                # Only update notes, keep status as "processing"
                new_notes = f"{speaker}: {progress_msg}"
                cursor.execute(
                    "UPDATE injections SET processing_notes = ? WHERE id = ?",
                    (new_notes, injection_id)
                )
                conn.commit()
                print(f"✅ Updated progress for {speaker} in database: {progress_msg}")
            conn.close()
        except Exception as e:
            print(f"⚠️ Error updating speaker status: {e}")

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
    
    # Update initial progress
    update_speaker_status(f"Starting audio generation for {len(speaker_lines)} lines")
    
    # Process each line for this speaker
    results = []
    for i, (line_idx, text) in enumerate(tqdm(speaker_lines, desc=f"Generating {speaker} audio")):
        # Progress update every few lines
        if i % 3 == 0 or i == len(speaker_lines) - 1:
            progress_pct = int((i + 1) / len(speaker_lines) * 100)
            update_speaker_status(f"Generating line {i+1}/{len(speaker_lines)} ({progress_pct}% complete)")
        
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
    
    # Final status update
    update_speaker_status(f"Completed audio generation for all {len(speaker_lines)} lines")
    print(f"✅ Completed all {len(speaker_lines)} lines for {speaker}")
    return results


@app.function(
    image=common_image,
    # No GPU needed for the coordinator
    container_idle_timeout=10*60,
    timeout=24*60*60,
    volumes={
        "/data": shared_volume,
        "/bark_models": bark_volume  # Add the Bark models volume
    } if bark_volume else {"/data": shared_volume},
    allow_concurrent_inputs=100,
)
def generate_audio(encoded_script: str, injection_id: str = None) -> str:
    """
    Takes the serialized script from generate_script() -> runs Bark TTS -> returns final .wav path
    Now with simplified path handling and multiple volume commits for reliability
    """
    # Set environment variables to use the pre-downloaded models if available
    if os.path.exists('/bark_models'):
        print("Using pre-downloaded Bark models from volume")
        os.environ["XDG_CACHE_HOME"] = "/bark_models"
    
    def update_status(status, notes=None):
        """Helper function to update database status and notes"""
        if not injection_id:
            return
            
        try:
            import sqlite3
            DB_PATH = "/data/injections_truncate.db"
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            if notes:
                cursor.execute(
                    "UPDATE injections SET status = ?, processing_notes = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (status, notes, injection_id)
                )
            else:
                cursor.execute(
                    "UPDATE injections SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (status, injection_id)
                )
            conn.commit()
            conn.close()
            print(f"✅ Updated status to '{status}' for ID: {injection_id}")
            if notes:
                print(f"📝 Notes: {notes}")
        except Exception as e:
            print(f"⚠️ Error updating database: {e}")
    
    print(f"🔊 Bark TTS starting with parallel processing. Received encoded script of size: {len(encoded_script)} bytes")
    
    # First status update - starting processing
    update_status("processing", "Preparing to generate audio...")

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

    try:
        # --- Step 1: Decode the serialized script ---
        update_status("processing", "Decoding script data...")
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
            
            update_status("processing", f"Preparing to generate audio for {len(lines)} dialogue lines")

            # Final safety check for script length
            if len(lines) > MAX_SCRIPT_LINES:
                print(f"⚠️ Script exceeds maximum length ({len(lines)} > {MAX_SCRIPT_LINES}), truncating...")
                truncated_lines = len(lines) - MAX_SCRIPT_LINES
                lines = lines[:MAX_SCRIPT_LINES]
                
                # Update processing notes
                update_status(
                    "processing", 
                    f"Script truncated from {len(script_data)} to {MAX_SCRIPT_LINES} lines (removed {truncated_lines} lines)"
                )
                print(f"Script truncated to {len(lines)} lines")

            # Calculate estimated processing time
            estimated_time = estimate_processing_time(lines)
            print(f"✨ Estimated processing time: {estimated_time} for {len(lines)} lines")
            update_status("processing", f"Estimated processing time: {estimated_time}")
            
        except Exception as e:
            print(f"❌ Error decoding serialized script: {e}")
            update_status("error", f"Error decoding script: {str(e)}")
            
            # Create a minimal fallback script
            lines = [
                ("Speaker 1", "Welcome to our podcast. Today we're discussing an interesting topic."),
                ("Speaker 2", "I'm excited to learn more about this. Could you tell us more?"),
                ("Speaker 1", "Of course! Let me explain some key points."),
                ("Speaker 2", "That's fascinating. What else should we know?")
            ]
            print("Using fallback script instead")
            update_status("processing", f"Using fallback script due to error: {str(e)}")

        # --- SPEAKER SEPARATION FOR PARALLEL PROCESSING ---
        update_status("processing", "Separating speaker dialogue for parallel processing...")
        
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

        print("\nDEBUG - Starting parallel processing of speakers:")
        # Process each speaker's lines in parallel
        speaker1_results = []
        speaker2_results = []

        # Only process if there are lines for that speaker
        if speaker1_lines:
            print(f"  Sending {len(speaker1_lines)} Speaker 1 lines to GPU #1")
            update_status("processing", f"Generating voice for Speaker 1 ({len(speaker1_lines)} lines)...")
            speaker1_results = generate_speaker_audio.remote(speaker1_lines, "Speaker 1", injection_id)

        if speaker2_lines:
            print(f"  Sending {len(speaker2_lines)} Speaker 2 lines to GPU #2")
            update_status("processing", f"Generating voice for Speaker 2 ({len(speaker2_lines)} lines)...")
            speaker2_results = generate_speaker_audio.remote(speaker2_lines, "Speaker 2", injection_id)

        # --- Combine results in original script order ---
        update_status("processing", "Speaker audio generated. Combining audio segments...")
        all_results = speaker1_results + speaker2_results
        all_results.sort(key=lambda x: x[0])  # Sort by original script line index

        print(f"\nDEBUG - Received {len(all_results)} total audio segments from parallel processing")
        print("  Recombining in original script order based on line indices...")

        # --- Assemble final audio ---
        update_status("processing", "Assembling final podcast audio...")
        segments = []
        rates = []
        turn_silence = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)  # Silence between turns
        
        for _, audio_array, sr in all_results:
            segments.append(audio_array)
            segments.append(turn_silence)  # Add half-second silence between turns
            rates.append(sr)
            rates.append(sr)  # Need matching rate for silence

        # --- Define standardized output path ---
        # Use a consistent file naming pattern based on injection_id
        final_audio_path = f"/data/podcast_audio/podcast_{injection_id}.wav"
        
        # --- Save to file ---
        update_status("processing", "Saving final podcast audio file...")
        final_clip = concatenate_audio_segments(segments, rates)
        final_clip.export(final_audio_path, format="wav", codec="pcm_s16le")  # Ensure proper WAV encoding

        # Early commit to ensure file is visible
        shared_volume.commit()
        
        # Update database with completed status and the final path
        update_status(
            "completed", 
            f"Generated podcast with {len(lines)} dialogue lines. Audio saved as podcast_{injection_id}.wav"
        )

        # Update database with file path information
        def update_database_path(injection_id, path):
            """Helper function to update the audio file path in database"""
            if not injection_id:
                return
                
            try:
                import sqlite3
                DB_PATH = "/data/injections_truncate.db"
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                cursor.execute(
                    "UPDATE injections SET processed_path = ? WHERE id = ?",
                    (path, injection_id)
                )
                conn.commit()
                conn.close()
                print(f"✅ Updated audio path in database for ID: {injection_id}")
            except Exception as e:
                print(f"⚠️ Error updating database path: {e}")

        # Update database with final audio path
        update_database_path(injection_id, final_audio_path)

        # Second explicit commit after database update
        shared_volume.commit()
        
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
        
    except Exception as e:
        error_message = f"Error during audio generation: {str(e)}"
        print(f"❌ {error_message}")
        update_status("error", error_message)
        raise e
