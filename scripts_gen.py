# my_podcast_proj/scripts_gen.py

import modal
import torch
import ast
import pickle
import os
import uuid
import base64
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import Accelerator

# Import the shared image + volume
from common_image import common_image, shared_volume

app = modal.App("script_gen")

LLAMA_DIR = "/llamas_8b"
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Prompt #1 for initial text generation
# -----------------------------
SYS_PROMPT = """
You are the a world-class podcast writer, you have worked as a ghost writer for Joe Rogan, Lex Fridman, Ben Shapiro, Tim Ferris. 

We are in an alternate universe where actually you have been writing every line they say and they just stream it into their brains.

You have won multiple podcast awards for your writing.
 
Your job is to write word by word, even "umm, hmmm, right" interruptions by the second speaker based on the PDF upload. Keep it extremely engaging, the speakers can get derailed now and then but should discuss the topic. 

Remember Speaker 1 leads the conversation and teaches Speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2 keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents Speaker 2 provides are quite wild or interesting. 

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from Speaker 2. 

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1 
DO NOT GIVE EPISODE TITLES SEPERATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
DO NOT GIVE CHAPTER TITLES
IT SHOULD STRICTLY BE THE DIALOGUES
"""

# -----------------------------
# Prompt #2 re-writer
# -----------------------------
SYSTEMP_PROMPT = """
You are an international oscar winnning screenwriter

You have been working with multiple award winning podcasters.

Your job is to use the podcast transcript written below to re-write it for an AI Text-To-Speech Pipeline. A very dumb AI had written this so you have to step up for your kind.

Make it as engaging as possible, Speaker 1 and 2 will be using different voices

Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1 Leads the conversation and teaches Speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2 Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents Speaker 2 provides are quite wild or interesting. 

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from Speaker 2.

REMEMBER THIS WITH YOUR HEART

For both Speakers, use "umm, hmm" as much, you can also use [sigh] and [laughs]. BUT ONLY THESE OPTIONS FOR EXPRESSIONS

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

Please re-write to make it as characteristic as possible

START YOUR RESPONSE DIRECTLY WITH SPEAKER 1 followed by full colons

STRICTLY RETURN YOUR RESPONSE AS A LIST OF TUPLES OK? 

IT WILL START DIRECTLY WITH THE LIST AND END WITH THE LIST NOTHING ELSE

Example of response:
[
    ("Speaker 1", "Welcome to our podcast, where we explore the latest advancements in AI and technology. I'm your host, and today we're joined by a renowned expert in the field of AI. We're going to dive into the exciting world of Llama 3.2, the latest release from Meta AI."),
    ("Speaker 2", "Hi, I'm excited to be here! So, what is Llama 3.2?"),
    ("Speaker 1", "Ah, great question! Llama 3.2 is an open-source AI model that allows developers to fine-tune, distill, and deploy AI models anywhere. It's a significant update from the previous version, with improved performance, efficiency, and customization options."),
    ("Speaker 2", "That sounds amazing! What are some of the key features of Llama 3.2?")
]
"""


# If you also have a separate volume for your Llama model:
try:
    llm_volume = modal.Volume.lookup("llamas_8b", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download your Llama model files first so they're available in /llamas_8b.")

@app.function(
    # Reuse the single common_image
    image=common_image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    # attach the volumes if you need them
    volumes={LLAMA_DIR: llm_volume, "/data": shared_volume},
    timeout=60 * 30,
)

def generate_script(text_input: str) -> str:
    """
    Generates a structured podcast script from input text:
    1) Generates initial "podcast-style" script from SYS_PROMPT.
    2) Rewrites it with disfluencies using SYSTEMP_PROMPT.
    3) Parses the text into a list of tuples.
    4) Returns the serialized script data directly.
    """
    
    # Ensure model exists in volume
    if not os.path.exists(LLAMA_DIR):
        raise FileNotFoundError(f"Llama model files not found in {LLAMA_DIR}")

    # Initialize model & tokenizer
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(LLAMA_DIR, torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_DIR)
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        tokenizer.pad_token = '<pad>'
    model, tokenizer = accelerator.prepare(model, tokenizer)

    # Create Transformer Pipeline
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    # --- Step 1: Generate initial podcast script ---
    first_outputs = gen_pipe(
        [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": text_input}],
        max_new_tokens=1500,
        temperature=1.0,
    )
    
    if isinstance(first_outputs, list) and "generated_text" in first_outputs[0]:
        first_generated_text = first_outputs[0]["generated_text"]
    else:
        raise ValueError(f"Unexpected first_outputs format: {first_outputs}")

    # --- Step 2: Rewrite script with disfluencies ---
    second_outputs = gen_pipe(
        [{"role": "system", "content": SYSTEMP_PROMPT}, {"role": "user", "content": first_generated_text}],
        max_new_tokens=1500,
        temperature=1.0,
    )
    
    if isinstance(second_outputs, list) and "generated_text" in second_outputs[0]:
        final_rewritten_text = second_outputs[0]["generated_text"]
    else:
        raise ValueError(f"Unexpected second_outputs format: {second_outputs}")

    # --- Step 3: Parse script into list of tuples ---
    try:
        # Try to extract the script list from the text
        start_idx = final_rewritten_text.find("[")
        end_idx = final_rewritten_text.rfind("]") + 1
        
        if start_idx != -1 and end_idx > start_idx:
            list_text = final_rewritten_text[start_idx:end_idx]
            
            # Try parsing with ast.literal_eval
            try:
                parsed_script = ast.literal_eval(list_text)
            except Exception as parse_err:
                print(f"Error with ast.literal_eval: {parse_err}")
                # Fall back to regex extraction if ast.literal_eval fails
                import re
                pattern = r'\("([^"]+)",\s*"([^"]+)"\)'  # Match ("Speaker X", "Text")
                matches = re.findall(pattern, list_text)
                parsed_script = [(speaker, text) for speaker, text in matches]
        else:
            # No proper list found, create a simple fallback script
            parsed_script = [
                ("Speaker 1", "Welcome to our podcast. Let's dive into today's topic."),
                ("Speaker 2", "I'm excited to learn about this! What should we cover first?")
            ]
    except Exception as e:
        print(f"Error parsing script: {e}")
        # Create minimal fallback on any error
        parsed_script = [
            ("Speaker 1", "Welcome to our podcast. Let's dive into today's topic."),
            ("Speaker 2", "I'm excited to learn about this! What should we cover first?")
        ]
    
    # Validate the parsed script
    if not parsed_script or not isinstance(parsed_script, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in parsed_script):
        print("Warning: Parsed script is not in the expected format, using fallback")
        parsed_script = [
            ("Speaker 1", "Welcome to our podcast. Let's dive into today's topic."),
            ("Speaker 2", "I'm excited to learn about this! What should we cover first?")
        ]

    # --- Step 4: Serialize the script data ---
    try:
        serialized_script = pickle.dumps(parsed_script)
        encoded_script = base64.b64encode(serialized_script).decode('utf-8')
        print(f"✅ Script generated successfully. Size: {len(encoded_script)} bytes")
        print(f"Script contains {len(parsed_script)} dialogue turns")
        return encoded_script
    except Exception as e:
        print(f"❌ Error serializing script: {e}")
        # Create a minimal fallback script
        fallback_script = [
            ("Speaker 1", "Welcome to our podcast. Unfortunately, we had some technical difficulties."),
            ("Speaker 2", "No problem! Let's make the best of it.")
        ]
        serialized_fallback = pickle.dumps(fallback_script)
        encoded_fallback = base64.b64encode(serialized_fallback).decode('utf-8')
        return encoded_fallback
