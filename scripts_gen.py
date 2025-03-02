import modal
import torch
import os
import ast
import pickle
import base64
from typing import Optional, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import Accelerator
from common_image import common_image, shared_volume
import re

# Define app
app = modal.App("scripts_gen")

# Constants
DATA_DIR = "/data"
SCRIPTS_FOLDER = "/data/podcast_scripts"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create necessary directories
os.makedirs(SCRIPTS_FOLDER, exist_ok=True)

# System prompts for script generation
SYSTEM_PROMPT = """
You are a world-class podcast writer, having ghostwritten for top shows like Joe Rogan, Lex Fridman, and Tim Ferris.
Your job is to write a lively, engaging script with two speakers based on the text I provide.
Speaker 1 leads the conversation, teaching Speaker 2, giving anecdotes and analogies.
Speaker 2 asks follow-up questions, gets excited or confused, and interrupts with "umm, hmm" occasionally.

ALWAYS START YOUR RESPONSE WITH 'SPEAKER 1' and a colon.
DO NOT GIVE SPEAKERS NAMES.
Keep the conversation extremely engaging, welcome the audience with a fun overview, etc.
Only create ONE EPISODE of the podcast. 
The speakers discussing a the topic as external commentators.
"""

REWRITE_PROMPT = """
You are an Oscar-winning screenwriter rewriting a transcript for an AI Text-To-Speech Pipeline.

Make it as engaging as possible, Speaker 1 and 2 will be using different voices.

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

Please re-write to make it as characteristic as possible

For both Speakers use the following disfluencies FREQUENTLY AS MUCH AS POSSIBLE, umm, hmm, [laughs], [sighs], [laughter], [gasps], [clears throat], — for hesitations, CAPITALIZATION for emphasis. BUT ONLY THESE OPTIONS FOR EXPRESSIONS

Return your final answer as a Python LIST of (Speaker, text) TUPLES ONLY, NO EXPLANATIONS, e.g.

Dont add "quotation marks" within the script dialogue. 

IT WILL START DIRECTLY WITH THE LIST AND END WITH THE LIST NOTHING ELSE

[
    ("Speaker 1", "Hello, and welcome..."),
    ("Speaker 2", "Hmm, that is fascinating!")
]

IMPORTANT Your response must be a valid Python list of tuples. STRICTLY RETURN YOUR RESPONSE AS A LIST OF TUPLES
"""

def clean_llm_output(text: str) -> str:
    """Remove prompt instructions from the LLM output"""
    # List of markers that indicate where the actual content starts
    content_markers = [
        'SPEAKER 1:', 
        'Speaker 1:',
        'Speaker 1,', 
        'SPEAKER 1 ,',  
        '[("Speaker 1"',
        '[("SPEAKER 1"'
    ]
    
    # Find the earliest occurrence of any marker
    start_index = float('inf')
    for marker in content_markers:
        idx = text.find(marker)
        if idx != -1 and idx < start_index:
            start_index = idx
    
    if start_index == float('inf'):
        return text
    
    return text[start_index:]


def convert_disfluencies(text):
    """
    Convert parenthesized expressions like (laughs) to bracketed [laughs]
    for proper TTS rendering.
    """
    disfluencies = [
        "laughs", "sighs", "laughter", "gasps", "clears throat", 
        "sigh", "laugh", "gasp",
        "hmm", "umm"
    ]
    
    # First check for parenthesized expressions
    for disfluency in disfluencies:
        # Convert (laughs) to [laughs]
        text = re.sub(r'\(' + disfluency + r'\)', '[' + disfluency + ']', text, flags=re.IGNORECASE)
        
        # Also convert <laughs> to [laughs] if present
        text = re.sub(r'<' + disfluency + r'>', '[' + disfluency + ']', text, flags=re.IGNORECASE)
    
    return text


def convert_disfluencies(text):
    """
    Convert parenthesized expressions like (laughs) to bracketed [laughs]
    for proper TTS rendering.
    """
    disfluencies = [
        "laughs", "sighs", "laughter", "gasps", "clears throat", 
        "sigh", "laugh", "gasp",
        "hmm", "umm"
    ]
    
    # First check for parenthesized expressions
    for disfluency in disfluencies:
        # Convert (laughs) to [laughs]
        text = re.sub(r'\(' + disfluency + r'\)', '[' + disfluency + ']', text, flags=re.IGNORECASE)
        
        # Also convert <laughs> to [laughs] if present
        text = re.sub(r'<' + disfluency + r'>', '[' + disfluency + ']', text, flags=re.IGNORECASE)
    
    return text

def parse_podcast_script(text: str) -> List[Tuple[str, str]]:
    """Parse the generated text into a list of speaker-text tuples"""
    # Clean the output first
    cleaned_text = clean_llm_output(text)
    
    try:
        # Extract everything between the first [ and last ]
        start_idx = cleaned_text.find("[")
        end_idx = cleaned_text.rfind("]") + 1
        if start_idx == -1 or end_idx == 0:
            # If no brackets found, try to parse line by line
            lines = cleaned_text.split('\n')
            script = []
            current_speaker = None
            current_text = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if ':' in line:
                    if current_speaker and current_text:
                        script.append((current_speaker, ' '.join(current_text)))
                        current_text = []
                    
                    parts = line.split(':', 1)
                    current_speaker = parts[0].strip()
                    if len(parts) > 1:
                        current_text.append(parts[1].strip())
                else:
                    if current_speaker:
                        current_text.append(line)
            
            if current_speaker and current_text:
                script.append((current_speaker, ' '.join(current_text)))
            
            return script if script else [("Speaker 1", cleaned_text)]
            
        # Try to parse the content between brackets
        candidate = cleaned_text[start_idx:end_idx]
        final_script = ast.literal_eval(candidate)
        
        if not isinstance(final_script, list):
            return [("Speaker 1", cleaned_text)]
            
        return final_script
        
    except Exception as e:
        print(f"Parsing error: {str(e)}")
        # Fallback to basic parsing if ast.literal_eval fails
        try:
            lines = cleaned_text.split('\n')
            script = []
            for line in lines:
                if ':' in line:
                    parts = line.split(':', 1)
                    speaker = parts[0].strip()
                    text = parts[1].strip() if len(parts) > 1 else ""
                    script.append((speaker, text))
            return script if script else [("Speaker 1", cleaned_text)]
        except Exception as e2:
            print(f"Fallback parsing error: {str(e2)}")
            return [("Speaker 1", cleaned_text)]

# Load Llama8B specific image
llama_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "transformers==4.46.1",
        "accelerate>=0.26.0",
        "torch>=2.0.0",
        "sentencepiece"
    )
)

# Look up Llama model volume
try:
    llm_volume = modal.Volume.lookup("llamas_8b", create_if_missing=False)
    LLAMA_DIR = "/llamas_8b"
    print("Llama 8B model volume found")
except modal.exception.NotFoundError:
    # Fall back to using the common image if Llama model not available
    llm_volume = None
    LLAMA_DIR = None
    print("Warning: 'llamas_8b' volume not found. Will use OpenAI API fallback if configured.")

@app.function(
    image=llama_image if LLAMA_DIR else common_image,
    gpu=modal.gpu.A100(count=1, size="80GB") if LLAMA_DIR else modal.gpu.T4(count=1),
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    volumes={LLAMA_DIR: llm_volume, "/data": shared_volume} if LLAMA_DIR else {"/data": shared_volume},
)
def generate_script(source_text: str) -> str:
    """
    Generates a podcast script from source text.
    Returns the base64 encoded pickled script (list of tuples).
    """
    print(f"🚀 Generating script from text of length: {len(source_text)} characters")
    
    if LLAMA_DIR:
        # Use local Llama model
        print("Loading Llama model...")
        accelerator = Accelerator()
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_DIR,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_DIR)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            
        model, tokenizer = accelerator.prepare(model, tokenizer)
        
        generation_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            max_new_tokens=1500,
            temperature=1.0,
        )

        # Generate initial script
        print("Generating initial script...")
        prompt_1 = SYSTEM_PROMPT + "\n\n" + source_text
        first_draft = generation_pipe(prompt_1)[0]["generated_text"]

        # Rewrite with disfluencies
        print("Adding natural speech patterns...")
        prompt_2 = REWRITE_PROMPT + "\n\n" + first_draft
        final_text = generation_pipe(prompt_2)[0]["generated_text"]

    else:
        # Add OpenAI fallback here if needed
        print("Using OpenAI API fallback (not implemented)")
        # This would be where you add OpenAI API call code
        final_text = f"""
        [
            ("Speaker 1", "Using OpenAI API fallback Welcome to our podcast! Today we're discussing an interesting topic from the provided content."),
            ("Speaker 2", "Hmm, sounds fascinating! What's it about?"),
            ("Speaker 1", "It's about {source_text[:100]}..."),
            ("Speaker 2", "Tell me more about that!")
        ]
        """

    # Parse into structured format
    final_script = parse_podcast_script(final_text)
    
    # Save the script to a file for debugging/reference
    import uuid
    file_uuid = uuid.uuid4().hex
    pkl_path = os.path.join(SCRIPTS_FOLDER, f"script_{file_uuid}.pkl")
    
    with open(pkl_path, "wb") as f:
        pickle.dump(final_script, f)
    
    print(f"Script saved to {pkl_path} with {len(final_script)} lines of dialogue")
    
    # Serialize the script for passing to the audio generation function
    serialized_data = pickle.dumps(final_script)
    encoded_data = base64.b64encode(serialized_data).decode('utf-8')
    
    return encoded_data
